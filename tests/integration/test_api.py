"""Tests for the inference API.

The API is exercised over HTTPX against the real ASGI application, so routing,
validation, serialisation and error handling are all covered by the same call a
client would make.

A stub model stands in for the registry so these tests stay fast and do not
depend on a promoted model existing. The stub is deliberately thin: it receives
the frame the real code built, which means schema-to-features wiring is still
under test even though scoring is not.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ml_platform.api.main import create_app
from ml_platform.api.model import FALLBACK_THRESHOLD, LoadedModel, ModelService
from ml_platform.api.schemas import APPLICATION_COLUMNS, LoanApplication
from ml_platform.config import Config

VALID_APPLICATION: dict[str, Any] = {
    "term_months": 84,
    "employees": 12,
    "jobs_created": 3,
    "jobs_retained": 8,
    "gross_approved": 250000.0,
    "sba_approved": 187500.0,
    "disbursed": 250000.0,
    "state": "CA",
    "bank_state": "CA",
    "revolving_line_of_credit": "N",
    "low_doc": "N",
    "urban_rural": 1,
    "new_business": 1,
    "naics": "722410",
    "franchise_code": 0,
    "approval_date": "2005-06-15",
    "disbursement_date": "2005-07-20",
}


class _StubPipeline:
    """Returns a fixed probability, and records what it was asked to score."""

    def __init__(self, probability: float = 0.42) -> None:
        self.probability = probability
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, features: pd.DataFrame) -> Any:
        import numpy as np

        self.seen = features
        return np.column_stack(
            [np.full(len(features), 1 - self.probability), np.full(len(features), self.probability)]
        )


def _loaded_model(probability: float = 0.42, threshold: float = 0.2) -> LoadedModel:
    return LoadedModel(
        pipeline=_StubPipeline(probability),
        name="sba-loan-default-classifier",
        version="1",
        alias="production",
        feature_set="engineered",
        mlflow_run_id="mlrun123",
        platform_run_id="candidate-20260906T175308Z",
        decision_threshold=threshold,
        threshold_source="registry",
    )


def _app(model: LoadedModel | None, error: str | None = None) -> FastAPI:
    """An app whose model service is pre-populated, so no registry is touched."""
    app = create_app(Config(raw={}, environment="test"), load_on_startup=False)
    service = ModelService(Config(raw={}, environment="test"))
    service._model = model
    service._error = error

    original = app.router.lifespan_context

    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(target: FastAPI) -> AsyncIterator[None]:
        async with original(target):
            target.state.model_service = service
            yield

    app.router.lifespan_context = lifespan
    return app


def _client(app: FastAPI) -> TestClient:
    """Starlette's client, which drives the ASGI app through HTTPX.

    Used as a context manager so the lifespan runs and the model service is
    installed on application state, exactly as it is when served.
    """
    return TestClient(app)


@pytest.fixture
def ready_app() -> FastAPI:
    return _app(_loaded_model())


@pytest.fixture
def unready_app() -> FastAPI:
    return _app(None, error="no model carries the 'production' alias")


class TestSchemaValidation:
    """Rejections happen before anything reaches the model."""

    def test_a_valid_application_parses(self) -> None:
        assert LoanApplication(**VALID_APPLICATION).term_months == 84

    def test_the_frame_carries_every_column_the_features_need(self) -> None:
        frame = LoanApplication(**VALID_APPLICATION).to_frame()
        assert list(frame.columns) == list(APPLICATION_COLUMNS)
        assert len(frame) == 1

    def test_dates_become_timestamps(self) -> None:
        frame = LoanApplication(**VALID_APPLICATION).to_frame()
        assert isinstance(frame["ApprovalDate"].iloc[0], pd.Timestamp)

    def test_a_stray_flag_code_is_normalised_not_rejected(self) -> None:
        """The register itself contains stray codes; training folded them to UNK."""
        frame = LoanApplication(**{**VALID_APPLICATION, "low_doc": "T"}).to_frame()
        assert frame["LowDoc"].iloc[0] == "UNK"

    def test_state_is_upper_cased(self) -> None:
        frame = LoanApplication(**{**VALID_APPLICATION, "state": "ca"}).to_frame()
        assert frame["State"].iloc[0] == "CA"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("term_months", 0),
            ("term_months", 700),
            ("gross_approved", 0),
            ("gross_approved", -1),
            ("employees", -1),
            ("state", "CALIF"),
            ("urban_rural", 7),
            ("new_business", 3),
            ("naics", "not-a-code"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError):
            LoanApplication(**{**VALID_APPLICATION, field: value})

    def test_a_guarantee_above_the_loan_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed"):
            LoanApplication(**{**VALID_APPLICATION, "sba_approved": 999_999.0})

    def test_disbursement_before_approval_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot precede"):
            LoanApplication(**{**VALID_APPLICATION, "disbursement_date": "2005-01-01"})

    def test_unknown_fields_are_rejected(self) -> None:
        """Silently ignoring a misspelled field would hide a caller's bug."""
        with pytest.raises(ValidationError):
            LoanApplication(**{**VALID_APPLICATION, "loan_amount": 1000})


class TestHealthAndReadiness:
    def test_health_answers_even_without_a_model(self, unready_app: FastAPI) -> None:
        """Liveness must not depend on a promoted model existing."""
        with _client(unready_app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_is_false_without_a_model(self, unready_app: FastAPI) -> None:
        with _client(unready_app) as client:
            response = client.get("/ready")
        payload = response.json()
        assert response.status_code == 200
        assert payload["status"] == "not_ready"
        assert payload["model_loaded"] is False
        assert "production" in payload["detail"]

    def test_readiness_is_true_with_a_model(self, ready_app: FastAPI) -> None:
        with _client(ready_app) as client:
            payload = client.get("/ready").json()
        assert payload["status"] == "ready"
        assert payload["model"]["version"] == "1"

    def test_model_metadata_identifies_the_served_model(self, ready_app: FastAPI) -> None:
        with _client(ready_app) as client:
            payload = client.get("/model").json()
        assert payload["name"] == "sba-loan-default-classifier"
        assert payload["alias"] == "production"
        assert payload["mlflow_run_id"] == "mlrun123"
        assert payload["platform_run_id"] == "candidate-20260906T175308Z"

    def test_model_metadata_is_unavailable_without_a_model(self, unready_app: FastAPI) -> None:
        with _client(unready_app) as client:
            assert client.get("/model").status_code == 503


class TestPrediction:
    def test_a_valid_application_is_scored(self, ready_app: FastAPI) -> None:
        with _client(ready_app) as client:
            response = client.post("/predict", json={"applications": [VALID_APPLICATION]})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["predictions"]) == 1
        assert payload["predictions"][0]["default_probability"] == pytest.approx(0.42)

    def test_the_flag_follows_the_threshold(self, ready_app: FastAPI) -> None:
        with _client(ready_app) as client:
            payload = client.post("/predict", json={"applications": [VALID_APPLICATION]}).json()
        # Probability 0.42 against a threshold of 0.2.
        assert payload["predictions"][0]["flagged"] is True
        assert payload["predictions"][0]["threshold"] == pytest.approx(0.2)

    def test_a_probability_below_the_threshold_is_not_flagged(self) -> None:
        app = _app(_loaded_model(probability=0.01, threshold=0.2))
        with _client(app) as client:
            payload = client.post("/predict", json={"applications": [VALID_APPLICATION]}).json()
        assert payload["predictions"][0]["flagged"] is False

    def test_every_response_identifies_the_model(self, ready_app: FastAPI) -> None:
        """A score is not traceable unless it says which model produced it."""
        with _client(ready_app) as client:
            payload = client.post("/predict", json={"applications": [VALID_APPLICATION]}).json()
        assert payload["model"]["version"] == "1"
        assert payload["model"]["feature_set"] == "engineered"

    def test_a_batch_is_scored_in_one_call(self, ready_app: FastAPI) -> None:
        applications = [VALID_APPLICATION, {**VALID_APPLICATION, "term_months": 120}]
        with _client(ready_app) as client:
            payload = client.post("/predict", json={"applications": applications}).json()
        assert len(payload["predictions"]) == 2

    def test_the_model_receives_engineered_features(self, ready_app: FastAPI) -> None:
        """Confirms the schema-to-features wiring, not just that a number came back."""
        with _client(ready_app) as client:
            client.post("/predict", json={"applications": [VALID_APPLICATION]})
        service = ready_app.state.model_service
        seen = service.model.pipeline.seen
        assert seen is not None
        assert "sba_guarantee_ratio" in seen.columns
        assert "naics_sector" in seen.columns

    def test_an_invalid_application_is_rejected_with_422(self, ready_app: FastAPI) -> None:
        payload = {"applications": [{**VALID_APPLICATION, "term_months": -5}]}
        with _client(ready_app) as client:
            response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_a_missing_required_field_is_rejected(self, ready_app: FastAPI) -> None:
        incomplete = {k: v for k, v in VALID_APPLICATION.items() if k != "gross_approved"}
        with _client(ready_app) as client:
            response = client.post("/predict", json={"applications": [incomplete]})
        assert response.status_code == 422
        assert "gross_approved" in response.text

    def test_an_empty_batch_is_rejected(self, ready_app: FastAPI) -> None:
        with _client(ready_app) as client:
            assert client.post("/predict", json={"applications": []}).status_code == 422

    def test_prediction_is_unavailable_without_a_model(self, unready_app: FastAPI) -> None:
        """Serving an unpromoted model would defeat the whole promotion system."""
        with _client(unready_app) as client:
            response = client.post("/predict", json={"applications": [VALID_APPLICATION]})
        assert response.status_code == 503

    def test_a_scoring_failure_returns_422_not_500(self) -> None:
        class _Broken:
            def predict_proba(self, _features: pd.DataFrame) -> Any:
                raise ValueError("feature mismatch")

        model = _loaded_model()
        model.pipeline = _Broken()
        with _client(_app(model)) as client:
            response = client.post("/predict", json={"applications": [VALID_APPLICATION]})
        assert response.status_code == 422
        assert "could not be scored" in response.json()["detail"]


class TestModelServiceIsolation:
    def test_the_model_is_loaded_once_not_per_request(self, ready_app: FastAPI) -> None:
        """Loading per request would add a registry round trip to every call."""
        with _client(ready_app) as client:
            client.post("/predict", json={"applications": [VALID_APPLICATION]})
            first = ready_app.state.model_service.model
            client.post("/predict", json={"applications": [VALID_APPLICATION]})
            second = ready_app.state.model_service.model
        assert first is second

    def test_an_unloaded_service_reports_its_reason(self) -> None:
        service = ModelService(Config(raw={}, environment="test"))
        assert not service.loaded
        assert service.error is None

    def test_a_missing_alias_leaves_the_service_unready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The service must never fall back to an unpromoted model."""
        from ml_platform.api import model as model_module

        monkeypatch.setattr(model_module, "resolve_production", lambda _c: None)
        config = Config(
            raw={"promotion": {"registered_model_name": "m", "production_alias": "production"}},
            environment="test",
        )
        service = ModelService(config)
        assert service.load() is False
        assert service.error is not None
        assert "production" in service.error


class TestThresholdSource:
    def test_a_registry_threshold_is_used_when_present(self) -> None:
        from ml_platform.api.model import _threshold_from

        value, source = _threshold_from({"validation_threshold_at_capacity": "0.0766"})
        assert value == pytest.approx(0.0766)
        assert source == "registry"

    def test_a_missing_threshold_falls_back_and_says_so(self) -> None:
        """A borrowed number must never be presented as the model's own."""
        from ml_platform.api.model import _threshold_from

        value, source = _threshold_from({})
        assert value == FALLBACK_THRESHOLD
        assert source == "fallback"

    def test_an_unparseable_threshold_falls_back(self) -> None:
        from ml_platform.api.model import _threshold_from

        assert _threshold_from({"validation_threshold_at_capacity": "n/a"})[1] == "fallback"
