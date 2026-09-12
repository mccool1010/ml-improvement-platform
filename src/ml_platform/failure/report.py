"""The evidence a scenario produces.

The shape is fixed on purpose, because the value of a failure exercise is almost
entirely in what it recorded. A run that says "passed" proves nothing later; one
that says what was broken, what was expected, what was seen, how far it spread,
what put it back and which invariant that demonstrated can be read by someone who
was not there.

``blast_radius`` is the field most worth filling in honestly. It is the answer to
"what else stopped working", and it is the difference between a platform that is
known to fail safely and one that merely has not failed yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_platform.failure.invariants import InvariantResult

LOGGER = logging.getLogger(__name__)

#: How a scenario was carried out. Reported, because evidence from a real
#: component being switched off is worth more than evidence from a stub, and a
#: reader is entitled to know which they are looking at.
MODE_LIVE = "live-kubernetes"
MODE_DOUBLE = "controlled-double"


@dataclass
class ScenarioEvidence:
    """One failure exercise, start to finish."""

    scenario: str
    description: str
    mode: str
    failure_injected: str
    expected_behaviour: str
    observed_behaviour: str = ""
    blast_radius: list[str] = field(default_factory=list)
    unaffected: list[str] = field(default_factory=list)
    recovery_action: str = ""
    recovery_result: str = ""
    recovered: bool = False
    invariants: list[InvariantResult] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    error: str | None = None

    @property
    def held(self) -> bool:
        """Whether every invariant this scenario claims to prove actually held."""
        return bool(self.invariants) and all(i.held for i in self.invariants)

    @property
    def passed(self) -> bool:
        return self.held and self.recovered and self.error is None

    def add(self, invariant: InvariantResult) -> InvariantResult:
        self.invariants.append(invariant)
        LOGGER.info("  %s", invariant.describe())
        return invariant

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC).isoformat()

    def summary(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        held = sum(1 for i in self.invariants if i.held)
        return (
            f"[{mark}] {self.scenario} ({self.mode}): {held}/{len(self.invariants)} "
            f"invariant(s) held, recovered={self.recovered}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "mode": self.mode,
            "passed": self.passed,
            "failure_injected": self.failure_injected,
            "expected_behaviour": self.expected_behaviour,
            "observed_behaviour": self.observed_behaviour,
            "blast_radius": self.blast_radius,
            "unaffected": self.unaffected,
            "recovery_action": self.recovery_action,
            "recovery_result": self.recovery_result,
            "recovered": self.recovered,
            "invariants": [i.to_dict() for i in self.invariants],
            "observations": self.observations,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


@dataclass
class FailureReport:
    """Every scenario from one run."""

    run_id: str
    scenarios: list[ScenarioEvidence] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return bool(self.scenarios) and all(s.passed for s in self.scenarios)

    def summary(self) -> str:
        passed = sum(1 for s in self.scenarios if s.passed)
        verdict = "PASS" if self.passed else "FAIL"
        return f"[{verdict}] {passed}/{len(self.scenarios)} scenario(s) passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "n_scenarios": len(self.scenarios),
            "n_passed": sum(1 for s in self.scenarios if s.passed),
            "invariants_checked": sorted({i.name for s in self.scenarios for i in s.invariants}),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.run_id}.json"
        destination.write_text(json.dumps(self.to_dict(), indent=2, default=str), "utf-8")
        LOGGER.info("wrote failure evidence to %s", destination)
        return destination


def new_run_id() -> str:
    return "failure-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
