"""FastAPI application.

The model is loaded once during startup and held on the application state, so
request handling never repeats that work.

A failure to load is not fatal. The service starts, ``/health`` answers, and
``/ready`` reports the reason it cannot serve. That distinction is what lets an
orchestrator hold traffic back from a service that is otherwise fine.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml_platform.api.model import ModelService
from ml_platform.api.platform import router as platform_router
from ml_platform.api.routes import router
from ml_platform.config import Config, load_config
from ml_platform.observability import API_SERVICE, metrics, tracing
from ml_platform.serving.canary import TIER_CANARY, TIER_PRODUCTION, CanaryRouter

LOGGER = logging.getLogger(__name__)


def create_app(config: Config | None = None, *, load_on_startup: bool = True) -> FastAPI:
    """Build the application.

    ``load_on_startup`` exists for tests that supply their own model service.
    """
    settings = config or load_config("production")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = ModelService(settings)
        if load_on_startup:
            service.load()
        app.state.model_service = service
        app.state.config = settings

        # The router reads its allocation from configuration, so every replica
        # agrees on the split by construction rather than by coordination.
        router = CanaryRouter()
        if settings.canary_enabled:
            router.start(
                traffic_percent=settings.canary_traffic_percent,
                candidate_url=str(settings.canary_predictor_url),
                candidate_version=service.model.canary_version if service.loaded else None,
                incumbent_version=service.model.version if service.loaded else None,
            )
        app.state.canary_router = router
        metrics.set_canary_traffic(router.traffic_percent)

        _publish_model_info(service)
        yield

    app = FastAPI(
        title="SBA loan default inference",
        description=(
            "Scores SBA loan applications with the model currently carrying the "
            "production alias in the MLflow registry."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    # Read-only aggregation for the M16 dashboard. A separate router so the
    # inference contract from M7 is untouched, and nothing here can serve a
    # model or change platform state.
    app.include_router(platform_router)
    _mount_dashboard(app)

    # Telemetry last, so it wraps the finished application. Both calls are
    # failure-tolerant: metrics never raise, and tracing returns False and logs
    # rather than preventing the service from starting.
    metrics.install(app, API_SERVICE)
    tracing.configure(app, API_SERVICE, endpoint=settings.otlp_endpoint)
    return app


def _mount_dashboard(app: FastAPI) -> None:
    """Serve the built dashboard, if it has been built.

    One serving surface rather than a second deployment: the dashboard is a
    presentation layer over this API, so it is served by the process that owns
    the data. Absent a build, nothing is mounted and the API is byte-for-byte
    what M7 through M15 shipped -- which is also why no test needs node
    installed.
    """
    from ml_platform.paths import project_root

    dist = project_root() / "dashboard" / "dist"
    if not (dist / "index.html").exists():
        LOGGER.info("no built dashboard at dashboard/dist; skipping the mount")
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/dashboard", StaticFiles(directory=str(dist), html=True), name="dashboard")

    from fastapi.responses import RedirectResponse

    # A visitor opening the bare address should land on something readable. Only
    # registered when the dashboard exists, so `/` stays unrouted otherwise.
    # HEAD too: hosting platforms probe the bare address with it.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard/")

    LOGGER.info("serving the dashboard from %s", dist)


def _publish_model_info(service: ModelService) -> None:
    """Put the resolved version on a gauge, so a graph can be attributed to it.

    The version label is the one deliberately broad label in the metric set:
    being able to say which promoted model a latency or error change belongs to
    is the reason the registry exists.
    """
    if not service.loaded:
        metrics.set_model_ready(API_SERVICE, False)
        return
    model = service.model
    metrics.set_model_ready(API_SERVICE, True)
    # Health per tier, which is what a canary decision reads first: an unhealthy
    # candidate is rolled back without measuring anything else.
    metrics.set_canary_tier_healthy(TIER_PRODUCTION, True)
    if model.canary_predictor is not None:
        reachable, _ = model.canary_predictor.ready()
        metrics.set_canary_tier_healthy(TIER_CANARY, reachable)
    metrics.set_model_info(API_SERVICE, model.name, model.version, model.alias, model.served_by)


app = create_app()
