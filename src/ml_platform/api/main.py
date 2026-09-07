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
    return app


app = create_app()
