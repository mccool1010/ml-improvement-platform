"""HTTP routes.

Handlers do three things: validate, delegate, shape a response. Feature building
and scoring live in :mod:`ml_platform.api.model`, so a request never triggers a
registry lookup or a deserialisation.

Health and readiness are deliberately different questions. Health asks whether
the process is alive; readiness asks whether it can actually score. A service
with no promoted model is healthy and unready, which is what lets an orchestrator
keep it running while withholding traffic.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response, status

from ml_platform.api.model import ModelNotLoadedError, ModelService
from ml_platform.api.schemas import (
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from ml_platform.observability import API_SERVICE
from ml_platform.observability.metrics import (
    OUTCOME_ERROR,
    OUTCOME_SUCCESS,
    OUTCOME_UNAVAILABLE,
    observe_prediction,
    set_model_ready,
)
from ml_platform.observability.tracing import add_span_attributes
from ml_platform.serving.client import PredictorError

LOGGER = logging.getLogger(__name__)

router = APIRouter()

SERVICE_NAME = "ml-platform-inference"


def _service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if not isinstance(service, ModelService):  # pragma: no cover - misbuilt app
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the model service is not configured",
        )
    return service


def _model_info(service: ModelService) -> ModelInfo:
    model = service.model
    return ModelInfo(
        name=model.name,
        version=model.version,
        alias=model.alias,
        feature_set=model.feature_set,
        mlflow_run_id=model.mlflow_run_id,
        platform_run_id=model.platform_run_id,
        decision_threshold=model.decision_threshold,
        threshold_source=model.threshold_source,
        served_by=model.served_by,
    )


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness. Answers as long as the process is serving."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["operations"],
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """Readiness. Reports whether a promoted model is loaded and can score.

    Not ready answers 503, not 200. An orchestrator decides whether to route
    traffic from the status code alone and never reads the body, so a readiness
    endpoint that always answered 200 would place a pod holding no model into
    the Service's endpoints. The body is unchanged either way, because a human
    reading it wants the reason.
    """
    service = _service(request)
    ready_now, detail = service.readiness()
    set_model_ready(API_SERVICE, ready_now)
    if not ready_now:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=service.loaded,
            detail=detail or "no production model is loaded",
            model=_model_info(service) if service.loaded else None,
        )
    return ReadinessResponse(status="ready", model_loaded=True, model=_model_info(service))


@router.get("/model", response_model=ModelInfo, tags=["operations"])
def model_metadata(request: Request) -> ModelInfo:
    """Which model is serving, and where it came from."""
    service = _service(request)
    try:
        return _model_info(service)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """Score one or more loan applications.

    Returns a default probability per application, and whether it clears the
    model's own decision threshold. Malformed applications are rejected by the
    schema before reaching this handler.
    """
    service = _service(request)
    try:
        model = service.model
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    import pandas as pd

    frame = pd.concat(
        [application.to_frame() for application in payload.applications], ignore_index=True
    )

    try:
        probabilities = model.predict(frame)
    except PredictorError as exc:
        observe_prediction(API_SERVICE, OUTCOME_UNAVAILABLE)
        # The model tier is unreachable or refused. The caller's request was
        # fine, so this is not 422: blaming a valid request for a dependency
        # failure sends the client looking in the wrong place, and a retry of
        # the identical request may well succeed.
        LOGGER.warning("the model tier could not be reached", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"the model tier could not be reached: {exc}",
        ) from exc
    except Exception as exc:
        observe_prediction(API_SERVICE, OUTCOME_ERROR)
        LOGGER.warning("scoring failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"the application could not be scored: {exc}",
        ) from exc

    threshold = model.decision_threshold
    predictions = [
        Prediction(
            default_probability=round(probability, 6),
            flagged=probability >= threshold,
            threshold=threshold,
        )
        for probability in probabilities
    ]

    observe_prediction(
        API_SERVICE,
        OUTCOME_SUCCESS,
        scored=len(predictions),
        flagged=sum(1 for p in predictions if p.flagged),
    )
    # Counts and identity only. The applications themselves are never put on a
    # span: what someone borrowed is not telemetry.
    add_span_attributes(
        **{
            "ml.batch.size": len(predictions),
            "ml.model.version": model.version,
            "ml.model.served_by": model.served_by,
        }
    )

    return PredictionResponse(predictions=predictions, model=_model_info(service))
