"""The KServe predictor: the model tier's HTTP server.

    uvicorn ml_platform.serving.predictor:app --host 0.0.0.0 --port 8080

Loads the model KServe placed at ``/mnt/models`` and scores register-shaped
application records. It speaks KServe's V1 protocol, so the endpoint is
``POST /v1/models/<name>:predict`` with an ``instances`` list.

Why this exists rather than ``mlflow models serve``. The registered artifact was
logged with the sklearn flavor's default, so its ``python_function`` flavor
records ``predict_fn: predict`` and MLflow's own scoring server therefore returns
hard labels. This project decides by comparing a probability against a
review-capacity threshold, so a label discards the number the decision is made
on. The alternative -- re-logging the model with ``pyfunc_predict_fn`` -- would
mint a new registry version and re-open a promotion decision that M6 already
made. Loading the sklearn flavor is what the project has always done; this
serves it, using the same :func:`ml_platform.serving.scoring.score`.

It holds no opinion about thresholds, flagging, or which version is production.
Those are the application tier's, in :mod:`ml_platform.api`. See ADR-002.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from ml_platform.observability import PREDICTOR_SERVICE, metrics, tracing
from ml_platform.serving.scoring import DEFAULT_FEATURE_SET, instances_to_frame, score

LOGGER = logging.getLogger(__name__)

#: Where KServe mounts the artifact. Overridable only for local testing.
ENV_MODEL_DIR = "MODEL_DIR"
#: The feature set the registered version records in its `feature_set` tag.
ENV_FEATURE_SET = "MODEL_FEATURE_SET"
#: Reported back so a response can be tied to the InferenceService that served it.
ENV_MODEL_NAME = "MODEL_NAME"

DEFAULT_MODEL_DIR = "/mnt/models"
DEFAULT_MODEL_NAME = "sba-loan-default"


class PredictRequest(BaseModel):
    """KServe V1 request: rows in the register's own column names."""

    instances: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    """KServe V1 response. One probability of default per instance."""

    predictions: list[float]


class ModelState:
    """The loaded pipeline, or the reason there is not one."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.error: str | None = None
        self.feature_set: str = os.environ.get(ENV_FEATURE_SET, DEFAULT_FEATURE_SET)
        self.name: str = os.environ.get(ENV_MODEL_NAME, DEFAULT_MODEL_NAME)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def load(self) -> bool:
        """Deserialise the model from the mounted directory.

        A failure is recorded rather than raised, for the same reason the API
        does it: a server that cannot load a model should start and report
        itself unready, so an orchestrator withholds traffic and the reason is
        visible, instead of a crash loop that says nothing.
        """
        directory = os.environ.get(ENV_MODEL_DIR, DEFAULT_MODEL_DIR)
        try:
            import mlflow.sklearn

            # The sklearn flavor, not pyfunc: this project scores with
            # predict_proba, which pyfunc does not expose for this artifact.
            self.pipeline = mlflow.sklearn.load_model(directory)
            self.error = None
            LOGGER.info("loaded model from %s (feature set %s)", directory, self.feature_set)
            return True
        except Exception as exc:
            self.pipeline = None
            self.error = f"could not load the model from {directory}: {exc}"
            LOGGER.warning(self.error, exc_info=True)
            return False


def create_app(state: ModelState | None = None, *, load_on_startup: bool = True) -> FastAPI:
    """Build the predictor application.

    ``load_on_startup`` exists for tests that supply their own state.
    """
    model_state = state or ModelState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_on_startup:
            model_state.load()
        app.state.model = model_state
        metrics.set_model_ready(PREDICTOR_SERVICE, model_state.ready)
        yield

    app = FastAPI(
        title="SBA loan default predictor",
        description="KServe model tier. Scores register-shaped applications.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def _state(request: Request) -> ModelState:
        held = getattr(request.app.state, "model", None)
        if not isinstance(held, ModelState):  # pragma: no cover - misbuilt app
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no model state")
        return held

    @app.get("/health", tags=["operations"])
    def health(request: Request, response: Response) -> dict[str, Any]:
        """Readiness, in the shape KServe's probes expect.

        503 when no model is loaded, so the pod stays out of the Service rather
        than accepting traffic it cannot serve.
        """
        held = _state(request)
        if not held.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"name": held.name, "ready": False, "detail": held.error}
        return {"name": held.name, "ready": True, "detail": None}

    @app.get("/v1/models/{name}", tags=["operations"])
    def model_ready(name: str, request: Request, response: Response) -> dict[str, Any]:
        """KServe V1 model-readiness."""
        held = _state(request)
        if not held.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"name": name, "ready": held.ready}

    @app.post("/v1/models/{name}:predict", response_model=PredictResponse, tags=["inference"])
    def predict(name: str, payload: PredictRequest, request: Request) -> PredictResponse:
        """KServe V1 predict. Probabilities, in the order the instances arrived."""
        held = _state(request)
        if name != held.name:
            # The V1 protocol addresses a named model. Answering for a name this
            # server does not serve would let a caller believe it had reached a
            # different model than it did.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"this server serves {held.name!r}, not {name!r}"
            )
        if not held.ready:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                held.error or "no model is loaded",
            )
        # instances_to_frame, not pd.DataFrame: the date columns arrive as
        # strings and must be coerced back before features are built.
        frame = instances_to_frame(payload.instances)
        try:
            probabilities = score(held.pipeline, frame, held.feature_set)
        except Exception as exc:
            # A frame the model cannot score is the caller's problem, not a
            # server fault; the application tier validates before it gets here.
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"the instances could not be scored: {exc}"
            ) from exc
        # Bounded, non-sensitive facts only. The applications themselves never
        # go into a span: a trace backend is not the place for someone's loan.
        tracing.add_span_attributes(
            **{
                "ml.model.name": held.name,
                "ml.model.feature_set": held.feature_set,
                "ml.batch.size": len(payload.instances),
            }
        )
        return PredictResponse(predictions=probabilities)

    metrics.install(app, PREDICTOR_SERVICE)
    tracing.configure(app, PREDICTOR_SERVICE)
    return app


app = create_app()
