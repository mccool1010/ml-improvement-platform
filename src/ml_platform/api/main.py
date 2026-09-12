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
from ml_platform.api.routes import router
from ml_platform.config import Config, load_config
from ml_platform.observability import API_SERVICE, metrics, tracing

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

    # Telemetry last, so it wraps the finished application. Both calls are
    # failure-tolerant: metrics never raise, and tracing returns False and logs
    # rather than preventing the service from starting.
    metrics.install(app, API_SERVICE)
    tracing.configure(app, API_SERVICE, endpoint=settings.otlp_endpoint)
    return app


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
    metrics.set_model_info(API_SERVICE, model.name, model.version, model.alias, model.served_by)


app = create_app()
