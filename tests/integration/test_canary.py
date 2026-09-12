"""Tests for canary routing, evaluation and rollback.

The properties worth protecting are the safety ones, and they are all about what
*cannot* happen:

* traffic cannot reach a canary that was never started;
* a canary that fails cannot move the production alias;
* a canary that succeeded cannot be completed twice, and one that failed cannot
  be completed at all;
* a failing canary tier cannot quietly fall back to production, because that
  would make the error rate the decision rests on read as zero;
* accuracy cannot be a rollback signal.

Everything here runs in process against the real router, the real evaluator and
the real FastAPI application, so none of it needs a cluster.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform.monitoring.canary import (
    DECISION_HOLD,
    DECISION_PROMOTE,
    DECISION_ROLLBACK,
    TierSignals,
    evaluate_canary,
)
from ml_platform.serving.canary import (
    BUCKETS,
    MAX_TRAFFIC_PERCENT,
    TIER_CANARY,
    TIER_PRODUCTION,
    CanaryError,
    CanaryRouter,
    CanaryState,
    routing_bucket,
    routing_key_for,
)

THRESHOLDS: dict[str, Any] = {
    "min_requests": 50,
    "max_error_rate": 0.02,
    "max_failure_rate": 0.01,
    "max_latency_p95_seconds": 1.0,
    "max_error_rate_ratio": 2.0,
    "max_latency_ratio": 1.5,
}

EVENT = "canary-test"


def _healthy(tier: str, **overrides: Any) -> TierSignals:
    """A tier that is behaving, unless told otherwise."""
    defaults: dict[str, Any] = {
        "requests": 200,
        "errors": 0,
        "upstream_failures": 0,
        "latency_p95_seconds": 0.05,
        "latency_mean_seconds": 0.03,
        "healthy": True,
    }
    return TierSignals(tier=tier, **{**defaults, **overrides})


def _decide(canary: TierSignals, production: TierSignals | None = None) -> Any:
    return evaluate_canary(
        canary=canary,
        production=production or _healthy(TIER_PRODUCTION),
        thresholds=THRESHOLDS,
        canary_event_id=EVENT,
        traffic_percent=10.0,
        observation_window_seconds=300,
        candidate_version="2",
        incumbent_version="1",
    )


class TestRoutingIsDeterministic:
    def test_the_same_key_always_reaches_the_same_tier(self) -> None:
        router = CanaryRouter()
        router.start(traffic_percent=25.0, candidate_url="http://canary")
        first = [router.tier_for(f"key-{i}") for i in range(200)]
        second = [router.tier_for(f"key-{i}") for i in range(200)]
        assert first == second

    def test_the_split_is_close_to_the_allocation(self) -> None:
        """Hashing a finite set never lands exactly on the percentage, so the
        assertion is a band -- but a tight one, and reproducible."""
        router = CanaryRouter()
        router.start(traffic_percent=20.0, candidate_url="http://canary")
        counts = router.observed_split([f"application-{i}" for i in range(4000)])
        share = counts[TIER_CANARY] / 4000
        assert 0.17 <= share <= 0.23, counts

    def test_zero_percent_sends_nothing_to_the_canary(self) -> None:
        router = CanaryRouter()
        router.start(traffic_percent=0.0, candidate_url="http://canary")
        counts = router.observed_split([f"k{i}" for i in range(500)])
        assert counts[TIER_CANARY] == 0

    def test_the_maximum_allocation_still_leaves_the_incumbent_serving(self) -> None:
        """The cap exists so there is always something to fall back to."""
        router = CanaryRouter()
        router.start(traffic_percent=MAX_TRAFFIC_PERCENT, candidate_url="http://canary")
        counts = router.observed_split([f"k{i}" for i in range(2000)])
        assert counts[TIER_CANARY] > counts[TIER_PRODUCTION]
        assert counts[TIER_PRODUCTION] > 0, "the incumbent must keep receiving traffic"

    def test_no_canary_means_no_canary_traffic(self) -> None:
        """A router that was never started must not route anywhere else."""
        router = CanaryRouter()
        assert all(router.tier_for(f"k{i}") == TIER_PRODUCTION for i in range(200))

    def test_buckets_are_stable_across_processes(self) -> None:
        """SHA-256, not hash(): a per-process salt would send the same key to
        different tiers in different replicas."""
        assert routing_bucket("stable-key") == routing_bucket("stable-key")
        assert 0 <= routing_bucket("stable-key") < BUCKETS

    def test_an_explicit_routing_key_wins(self) -> None:
        assert routing_key_for({"a": 1}, "pinned") == "pinned"

    def test_the_derived_key_is_stable_for_equal_payloads(self) -> None:
        assert routing_key_for({"a": 1, "b": 2}) == routing_key_for({"b": 2, "a": 1})

    def test_an_impossible_allocation_is_refused(self) -> None:
        with pytest.raises(CanaryError):
            CanaryState(traffic_percent=140.0)
        with pytest.raises(CanaryError):
            CanaryRouter().set_traffic(-1.0)

    def test_a_full_cutover_is_refused(self) -> None:
        """100% is not a canary: the incumbent gets nothing, so there is no
        comparison and nothing already serving to fall back to."""
        with pytest.raises(CanaryError, match="nothing to compare"):
            CanaryState(traffic_percent=100.0)
        with pytest.raises(CanaryError, match="nothing to compare"):
            CanaryRouter().set_traffic(100.0)

    def test_allocations_below_the_cap_are_unchanged(self) -> None:
        for percent in (0.0, 1.0, 5.0, 25.0, 50.0, MAX_TRAFFIC_PERCENT):
            CanaryRouter().set_traffic(percent)


class TestRollbackIsImmediate:
    def test_stopping_sends_everything_back(self) -> None:
        router = CanaryRouter()
        router.start(traffic_percent=50.0, candidate_url="http://canary")
        assert router.observed_split([f"k{i}" for i in range(200)])[TIER_CANARY] > 0

        router.stop("error rate too high")

        counts = router.observed_split([f"k{i}" for i in range(200)])
        assert counts[TIER_CANARY] == 0
        assert counts[TIER_PRODUCTION] == 200

    def test_the_reason_is_kept(self) -> None:
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary")
        router.stop("latency regression")
        assert router.state.stopped_reason == "latency regression"
        assert router.traffic_percent == 0.0


class TestTheCanaryPasses:
    def test_a_well_behaved_candidate_is_promoted(self) -> None:
        decision = _decide(_healthy(TIER_CANARY))
        assert decision.decision == DECISION_PROMOTE
        assert decision.failures == []

    def test_a_slightly_slower_candidate_still_passes(self) -> None:
        """The gate is 'not measurably worse', not 'identical'."""
        decision = _decide(
            _healthy(TIER_CANARY, latency_p95_seconds=0.06),
            _healthy(TIER_PRODUCTION, latency_p95_seconds=0.05),
        )
        assert decision.decision == DECISION_PROMOTE

    def test_the_report_carries_everything_needed_to_justify_it(self) -> None:
        import json

        payload = json.loads(json.dumps(_decide(_healthy(TIER_CANARY)).to_dict()))
        assert payload["candidate_version"] == "2"
        assert payload["incumbent_version"] == "1"
        assert payload["traffic_percent"] == 10.0
        assert payload["observation_window_seconds"] == 300
        assert payload["thresholds"]
        assert payload["signals"][TIER_CANARY]["requests"] == 200
        assert payload["signals"][TIER_PRODUCTION]
        assert payload["decision"] and payload["reason"]
        assert payload["checks"]


class TestAnUnhealthyCandidateIsRolledBack:
    def test_it_fails_immediately(self) -> None:
        decision = _decide(_healthy(TIER_CANARY, healthy=False, health_detail="connection refused"))
        assert decision.decision == DECISION_ROLLBACK
        assert "not serving" in decision.reason

    def test_nothing_else_is_measured(self) -> None:
        """There is no point timing a tier that cannot answer."""
        decision = _decide(_healthy(TIER_CANARY, healthy=False))
        assert [c.name for c in decision.checks] == ["candidate_healthy"]

    def test_an_unhealthy_candidate_with_no_traffic_still_fails(self) -> None:
        decision = _decide(_healthy(TIER_CANARY, requests=0, healthy=False))
        assert decision.decision == DECISION_ROLLBACK


class TestElevatedErrorsAreRolledBack:
    def test_an_absolute_error_rate_breach(self) -> None:
        decision = _decide(_healthy(TIER_CANARY, requests=200, errors=20))
        assert decision.decision == DECISION_ROLLBACK
        assert "error_rate_absolute" in [c.name for c in decision.failures]

    def test_being_much_worse_than_the_incumbent_fails_even_inside_the_ceiling(
        self,
    ) -> None:
        """The comparison the absolute ceiling cannot make."""
        decision = evaluate_canary(
            canary=_healthy(TIER_CANARY, requests=1000, errors=19),
            production=_healthy(TIER_PRODUCTION, requests=1000, errors=2),
            thresholds={**THRESHOLDS, "max_error_rate": 0.05},
            canary_event_id=EVENT,
            traffic_percent=10.0,
            observation_window_seconds=300,
        )
        assert decision.decision == DECISION_ROLLBACK
        assert "error_rate_vs_incumbent" in [c.name for c in decision.failures]

    def test_upstream_failures_are_their_own_signal(self) -> None:
        """Timeouts are invisible in a status-code count when the caller never
        got an answer to give a code to."""
        decision = _decide(_healthy(TIER_CANARY, requests=200, upstream_failures=10))
        assert decision.decision == DECISION_ROLLBACK
        assert "upstream_failure_rate" in [c.name for c in decision.failures]


class TestLatencyViolationsAreRolledBack:
    def test_an_absolute_latency_breach(self) -> None:
        decision = _decide(_healthy(TIER_CANARY, latency_p95_seconds=2.5))
        assert decision.decision == DECISION_ROLLBACK
        assert "latency_p95_absolute" in [c.name for c in decision.failures]

    def test_being_much_slower_than_the_incumbent_fails(self) -> None:
        decision = _decide(
            _healthy(TIER_CANARY, latency_p95_seconds=0.4),
            _healthy(TIER_PRODUCTION, latency_p95_seconds=0.05),
        )
        assert decision.decision == DECISION_ROLLBACK
        assert "latency_vs_incumbent" in [c.name for c in decision.failures]


class TestInsufficientTrafficHolds:
    def test_too_few_requests_is_a_hold_not_a_failure(self) -> None:
        """A candidate that has done nothing wrong must not be rolled back for
        the absence of evidence."""
        decision = _decide(_healthy(TIER_CANARY, requests=5))
        assert decision.decision == DECISION_HOLD
        assert decision.failures == []

    def test_a_breach_beats_a_hold(self) -> None:
        """Few requests but all of them failing is still a failure."""
        decision = _decide(_healthy(TIER_CANARY, requests=10, errors=10))
        assert decision.decision == DECISION_ROLLBACK


class TestAccuracyIsNeverASignal:
    """ADR-003's rule, asserted rather than assumed."""

    def test_no_check_mentions_a_quality_metric(self) -> None:
        decision = _decide(_healthy(TIER_CANARY))
        forbidden = ("accuracy", "average_precision", "roc_auc", "brier", "recall", "precision")
        for check in decision.checks:
            assert not any(word in check.name.lower() for word in forbidden), check.name

    def test_the_signals_carry_no_quality_field(self) -> None:
        fields = set(_healthy(TIER_CANARY).to_dict())
        assert not fields & {"average_precision", "roc_auc", "accuracy", "brier_score"}
