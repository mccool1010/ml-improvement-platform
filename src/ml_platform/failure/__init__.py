"""Deliberate failure injection, and the evidence it produces.

Not a chaos framework. Six named scenarios, each breaking one real thing and
checking one named property, with a report that records what spread and what did
not. The point is not that failures are survivable in general; it is that these
specific failures have these specific blast radii, and that the properties in
:mod:`ml_platform.failure.invariants` held while they happened.

Everything that breaks something is restored in a ``finally``.
"""

from ml_platform.failure import cluster, invariants, report, scenarios

__all__ = ["cluster", "invariants", "report", "scenarios"]
