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
import time

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
    observe_canary_request,
    observe_prediction,
    set_model_ready,
)
from ml_platform.observability.tracing import add_span_attributes
from ml_platform.serving.canary import (
    ROUTING_KEY_HEADER,
    TIER_CANARY,
    TIER_PRODUCTION,
    CanaryRouter,
    routing_key_for,
)
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


def _router(request: Request) -> CanaryRouter:
    held = getattr(request.app.state, "canary_router", None)
    return held if isinstance(held, CanaryRouter) else CanaryRouter()


def _model_info(service: ModelService, tier: str = TIER_PRODUCTION) -> ModelInfo:
    """Which model answered.

    When the canary served the request, the reported version is the candidate's
    -- reporting the incumbent's would make a canary untraceable in exactly the
    situation where tracing matters.
    """
    model = service.model
    if tier == TIER_CANARY and model.canary_predictor is not None:
        return ModelInfo(
            name=model.name,
            version=model.canary_version,
            alias="canary",
            feature_set=model.feature_set,
            mlflow_run_id=model.mlflow_run_id,
            platform_run_id=model.platform_run_id,
            decision_threshold=model.decision_threshold,
            threshold_source=model.threshold_source,
            served_by=model.served_by,
            serving_tier=TIER_CANARY,
        )
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

    # Which tier serves this request. Deterministic and sticky: the same
    # application always reaches the same model, so a retry cannot silently
    # cross tiers and a report can name exactly which requests it covered.
    router = _router(request)
    routing_key = routing_key_for(
        payload.model_dump(mode="json"), request.headers.get(ROUTING_KEY_HEADER)
    )
    tier = router.tier_for(routing_key)
    started = time.perf_counter()

    try:
        probabilities = model.predict_with(tier, frame)
    except PredictorError as exc:
        observe_canary_request(
            tier,
            duration_seconds=time.perf_counter() - started,
            failed=True,
            upstream_failure=True,
        )
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
        observe_canary_request(tier, duration_seconds=time.perf_counter() - started, failed=True)
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

    observe_canary_request(tier, duration_seconds=time.perf_counter() - started)
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
            "ml.serving.tier": tier,
        }
    )

    return PredictionResponse(predictions=predictions, model=_model_info(service, tier=tier))
