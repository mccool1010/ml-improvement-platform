"""Tests for the model tier and the two paths that reach it.

ADR-002 leaves the project with one scoring implementation reached two ways: in
process, when the API holds the model, and over HTTP, when a KServe
InferenceService does. The consequence the ADR names is that "that difference is
real and must be covered by an integration test, not assumed away".

The tests that matter here are the ones about the seam. A wire format whose two
halves disagree, or a tier that answers for a model it does not serve, produces
a wrong number rather than an error, and nothing else in the project would
notice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_platform.api.schemas import LoanApplication
from ml_platform.serving.client import PredictorError, RemotePredictor
from ml_platform.serving.predictor import ModelState, create_app
from ml_platform.serving.scoring import (
    DATE_COLUMNS,
    frame_to_instances,
    instances_to_frame,
    score,
)

APPLICATION: dict[str, Any] = {
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

SECOND_APPLICATION = {
    **APPLICATION,
    "term_months": 60,
    "state": "FL",
    "bank_state": "NY",
    "revolving_line_of_credit": "Y",
    "urban_rural": 2,
    "new_business": 2,
    "gross_approved": 35000.0,
    "sba_approved": 17500.0,
    "disbursed": 35000.0,
}


def _frame(*applications: dict[str, Any]) -> pd.DataFrame:
    records = applications or (APPLICATION,)
    return pd.concat(
        [LoanApplication(**record).to_frame() for record in records], ignore_index=True
    )


class _StubPipeline:
    """Scores from one numeric column, so a mangled frame changes the answer."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, features: pd.DataFrame) -> Any:
        self.seen = features
        # Anything derived from the data will do; the point is that it is not
        # constant, so a frame that lost information cannot silently agree.
        positive = (features["Term"].to_numpy(dtype=float) % 97) / 1000.0
        return np.column_stack([1 - positive, positive])


class TestTheWireFormatRoundTrips:
    """The two halves are a pair. If they disagree, the served feature
    representation stops matching the trained one without anything failing."""

    def test_the_values_survive(self) -> None:
        frame = _frame(APPLICATION, SECOND_APPLICATION)
        restored = instances_to_frame(frame_to_instances(frame))[frame.columns]
        for column in frame.columns:
            assert list(frame[column]) == list(restored[column]), column

    def test_dates_come_back_as_datetimes(self) -> None:
        """The engineered features take a day difference and read a month off a
        date. Left as text, both raise rather than answering wrongly -- but only
        if the coercion happens at all."""
        restored = instances_to_frame(frame_to_instances(_frame()))
        for column in DATE_COLUMNS:
            assert pd.api.types.is_datetime64_any_dtype(restored[column]), column

    def test_instances_are_json_safe(self) -> None:
        import json

        json.dumps(frame_to_instances(_frame(APPLICATION, SECOND_APPLICATION)))

    def test_scoring_is_unchanged_by_the_round_trip(self) -> None:
        """The property that actually matters: the number does not move."""
        frame = _frame(APPLICATION, SECOND_APPLICATION)
        pipeline = _StubPipeline()
        direct = score(pipeline, frame)
        through_the_wire = score(pipeline, instances_to_frame(frame_to_instances(frame)))
        assert direct == through_the_wire


@pytest.fixture
def predictor_client() -> Any:
    """The model tier, with a stub pipeline in place of the real artifact."""
    state = ModelState()
    state.pipeline = _StubPipeline()
    state.name = "sba-loan-default"
    with TestClient(create_app(state, load_on_startup=False)) as client:
        yield client


@pytest.fixture
def unready_predictor_client() -> Any:
    state = ModelState()
    state.error = "could not load the model from /mnt/models: no such directory"
    with TestClient(create_app(state, load_on_startup=False)) as client:
        yield client


class TestThePredictor:
    def test_a_loaded_model_reports_ready(self, predictor_client: Any) -> None:
        response = predictor_client.get("/v1/models/sba-loan-default")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_an_unloaded_model_answers_503(self, unready_predictor_client: Any) -> None:
        """Readiness must fail closed. A 200 here would put a predictor holding
        no model into the Service's endpoints."""
        response = unready_predictor_client.get("/health")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_it_scores_instances(self, predictor_client: Any) -> None:
        payload = {"instances": frame_to_instances(_frame(APPLICATION, SECOND_APPLICATION))}
        response = predictor_client.post("/v1/models/sba-loan-default:predict", json=payload)
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 2
        assert all(0.0 <= p <= 1.0 for p in predictions)

    def test_it_receives_engineered_features(self, predictor_client: Any) -> None:
        """The model tier builds features itself. If the application tier did it
        instead, the two could drift apart across a deployment."""
        payload = {"instances": frame_to_instances(_frame())}
        predictor_client.post("/v1/models/sba-loan-default:predict", json=payload)
        seen = predictor_client.app.state.model.pipeline.seen
        assert seen is not None
        # Engineered columns are present and the raw dates have been consumed.
        assert "term_bucket" in seen.columns
        assert "same_state_lender" in seen.columns
        assert "ApprovalDate" not in seen.columns

    def test_it_refuses_a_model_name_it_does_not_serve(self, predictor_client: Any) -> None:
        """V1 addresses a named model. Answering anyway would let a caller
        believe it had reached a different model than it did."""
        payload = {"instances": frame_to_instances(_frame())}
        response = predictor_client.post("/v1/models/some-other-model:predict", json=payload)
        assert response.status_code == 404

    def test_it_refuses_to_score_without_a_model(self, unready_predictor_client: Any) -> None:
        payload = {"instances": frame_to_instances(_frame())}
        response = unready_predictor_client.post(
            "/v1/models/sba-loan-default:predict", json=payload
        )
        assert response.status_code == 503

    def test_an_empty_instance_list_is_rejected(self, predictor_client: Any) -> None:
        response = predictor_client.post(
            "/v1/models/sba-loan-default:predict", json={"instances": []}
        )
        assert response.status_code == 422

    def test_unscoreable_instances_are_a_client_error(self, predictor_client: Any) -> None:
        """Not a 500. The application tier validates first, so a frame that
        reaches here and cannot be scored is the caller's problem."""
        response = predictor_client.post(
            "/v1/models/sba-loan-default:predict", json={"instances": [{"nonsense": 1}]}
        )
        assert response.status_code == 400


class TestTheClientAndPredictorAgree:
    """End to end across the seam, with the client talking to the real app."""

    def test_a_prediction_matches_scoring_the_frame_directly(self, predictor_client: Any) -> None:
        frame = _frame(APPLICATION, SECOND_APPLICATION)
        pipeline = predictor_client.app.state.model.pipeline

        response = predictor_client.post(
            "/v1/models/sba-loan-default:predict",
            json={"instances": frame_to_instances(frame)},
        )
        remote = response.json()["predictions"]
        local = score(pipeline, frame)
        assert remote == pytest.approx(local, abs=0.0)

    def test_the_client_builds_the_v1_paths(self) -> None:
        predictor = RemotePredictor("http://sba-loan-default-predictor/", "sba-loan-default")
        assert predictor.predict_url == (
            "http://sba-loan-default-predictor/v1/models/sba-loan-default:predict"
        )
        assert predictor.ready_url == "http://sba-loan-default-predictor/v1/models/sba-loan-default"

    def test_an_unreachable_predictor_raises_rather_than_falling_back(self) -> None:
        """No silent fallback to in-process scoring. Two things that could answer
        the same request leave nobody able to say which one did."""
        predictor = RemotePredictor("http://127.0.0.1:1", "sba-loan-default", timeout=0.25)
        with pytest.raises(PredictorError, match="could not reach"):
            predictor.predict(_frame())

    def test_an_unreachable_predictor_is_not_ready(self) -> None:
        predictor = RemotePredictor("http://127.0.0.1:1", "sba-loan-default", timeout=0.25)
        ready, reason = predictor.ready()
        assert ready is False
        assert reason is not None and "could not reach" in reason


class TestTheApplicationTierReadinessFollowsTheModelTier:
    """When KServe holds the model, this service holds none -- only the registry
    metadata it resolved at startup, and a dependency on the model tier."""

    def _service(self, predictor: Any) -> Any:
        from ml_platform.api.model import LoadedModel, ModelService

        service = ModelService.__new__(ModelService)
        service._config = None
        service._error = None
        service._model = LoadedModel(
            pipeline=None,
            name="sba-loan-default-classifier",
            version="1",
            alias="production",
            feature_set="engineered",
            mlflow_run_id="abc",
            platform_run_id="candidate-1",
            decision_threshold=0.5,
            threshold_source="fallback",
            predictor=predictor,
        )
        return service

    def test_it_is_ready_when_the_model_tier_is(self) -> None:
        class _Up:
            def ready(self) -> tuple[bool, str | None]:
                return True, None

        ready, detail = self._service(_Up()).readiness()
        assert ready is True
        assert detail is None

    def test_it_is_not_ready_when_the_model_tier_is_not(self) -> None:
        """Reporting ready while the thing that actually scores is gone would
        make every request fail after traffic had already been routed here."""

        class _Down:
            def ready(self) -> tuple[bool, str | None]:
                return False, "connection refused"

        ready, detail = self._service(_Down()).readiness()
        assert ready is False
        assert detail is not None and "model tier is not ready" in detail

    def test_it_is_asked_again_each_time(self) -> None:
        """Not a snapshot from startup. Otherwise the two Deployments would
        depend on the order they happened to start in."""

        class _Flaky:
            def __init__(self) -> None:
                self.answers = [(False, "starting"), (True, None)]

            def ready(self) -> tuple[bool, str | None]:
                return self.answers.pop(0)

        service = self._service(_Flaky())
        assert service.readiness()[0] is False
        assert service.readiness()[0] is True

    def test_reporting_the_model_tier_serves(self) -> None:
        from ml_platform.api.model import LoadedModel

        model = LoadedModel(
            pipeline=None,
            name="n",
            version="1",
            alias="production",
            feature_set="engineered",
            mlflow_run_id=None,
            platform_run_id=None,
            decision_threshold=0.5,
            threshold_source="fallback",
            predictor=RemotePredictor("http://x", "m"),
        )
        assert model.served_by == "kserve"
        model.predictor = None
        assert model.served_by == "in-process"


class TestAnUnreachableModelTierIsNotTheCallersFault:
    def test_it_answers_503_not_422(self) -> None:
        """A valid request that fails because a dependency is down must not be
        reported as a bad request: it sends the caller looking in the wrong
        place, and the identical request may succeed on retry."""
        from fastapi import FastAPI

        from ml_platform.api.main import create_app
        from ml_platform.api.model import LoadedModel, ModelService

        class _Down:
            def ready(self) -> tuple[bool, str | None]:
                return True, None

            def predict(self, frame: Any) -> list[float]:
                raise PredictorError(f"could not reach the predictor at http://x ({len(frame)})")

        service = ModelService.__new__(ModelService)
        service._config = None
        service._error = None
        service._model = LoadedModel(
            pipeline=None,
            name="sba-loan-default-classifier",
            version="1",
            alias="production",
            feature_set="engineered",
            mlflow_run_id=None,
            platform_run_id=None,
            decision_threshold=0.5,
            threshold_source="fallback",
            predictor=_Down(),
        )

        app: FastAPI = create_app(load_on_startup=False)
        with TestClient(app) as client:
            client.app.state.model_service = service
            response = client.post("/predict", json={"applications": [APPLICATION]})
        assert response.status_code == 503
        assert "model tier" in response.json()["detail"]
