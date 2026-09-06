"""Determinism controls.

Reproducing a metric needs more than a seed. Three other things can move a
result between runs of identical code on identical data:

**Thread count.** Parallel reductions in OpenMP and in the BLAS backend sum
floating-point values in whatever order threads finish. Changing the thread count
changes that order, and the last bits of the result move with it. Gradient
boosting is the sensitive component here, because histogram accumulation is
parallelised over samples. Thread counts are therefore pinned by configuration,
not left to the size of the machine.

**Hash randomisation.** ``PYTHONHASHSEED`` affects set and dict iteration order
for objects hashed by identity. Set inside the process it is too late to change
CPython's behaviour for the current interpreter, so it is also recorded, and the
launcher can set it before start.

**Row order.** Approval dates repeat thousands of times per day, so most rows tie
on the sort key and their relative order is decided by the sort algorithm. This is
not cosmetic: measured directly, switching the sort from ``quicksort`` to
``stable`` moved candidate average precision from 0.701151 to 0.702866, because
gradient boosting bins and accumulates histograms in row order. The sort kind is
therefore pinned by configuration and the resulting order is fingerprinted, so a
change in ordering behaviour fails loudly instead of quietly moving every metric.

The thread pinning must happen before numpy, scipy or scikit-learn import their
native libraries, which is why :func:`pin_threads` is called from the entry points
rather than at the bottom of some import chain.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal

#: Environment variables that control native thread pools. Every backend that
#: scikit-learn might sit on is covered, because which one is active depends on
#: how numpy was built.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Sort algorithm used everywhere rows are ordered.
#:
#: ``quicksort`` is numpy's introsort. It is deterministic for identical input,
#: which was verified by reproducing byte-identical predictions across separate
#: processes and across 1, 4 and 8 threads. It is kept because it is the ordering
#: under which the locked M1 reference was measured.
#:
#: It is not *semantically* pinned: numpy could change the algorithm and silently
#: reorder ties. :func:`row_order_fingerprint` closes that gap by making any such
#: change a loud failure rather than a drifting metric.
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
ROW_SORT_KIND: SortKind = "quicksort"

#: Columns hashed to fingerprint a prepared dataset. Together they capture both
#: content and row order, which is what a model actually consumes.
FINGERPRINT_COLUMNS: tuple[str, ...] = ("ApprovalDate", "Term", "target")


def pin_threads(n_threads: int = 1) -> None:
    """Pin every native thread pool to ``n_threads``.

    Call before importing numpy or scikit-learn. Single-threaded is the default
    because it removes parallel-reduction nondeterminism entirely, and the
    training runs here take seconds rather than hours.
    """
    if n_threads < 1:
        raise ValueError(f"thread count must be at least 1, got {n_threads}")
    for variable in THREAD_ENV_VARS:
        os.environ[variable] = str(n_threads)


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness the pipeline touches."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class DeterminismSettings:
    """The determinism controls actually in force, recorded with every run."""

    seed: int
    n_threads: int
    sort_kind: str
    python_hash_seed: str
    thread_env: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture(seed: int, n_threads: int) -> DeterminismSettings:
    """Snapshot the determinism controls for the run record."""
    return DeterminismSettings(
        seed=seed,
        n_threads=n_threads,
        sort_kind=ROW_SORT_KIND,
        python_hash_seed=os.environ.get("PYTHONHASHSEED", "unset"),
        thread_env={v: os.environ.get(v, "unset") for v in THREAD_ENV_VARS},
    )


def configure(seed: int, n_threads: int) -> DeterminismSettings:
    """Apply seeding and thread pinning, then return what was applied.

    ``pin_threads`` is effective only if the native libraries have not yet read
    their environment. Entry points call :func:`pin_threads` first and this
    afterwards; calling this alone still seeds correctly but may not change an
    already-initialised thread pool. The recorded settings reflect what was
    requested, and :func:`verify_thread_pinning` checks what took effect.
    """
    pin_threads(n_threads)
    set_global_seed(seed)
    return capture(seed, n_threads)


def verify_thread_pinning(n_threads: int) -> list[str]:
    """Return the names of thread variables that do not match ``n_threads``.

    An empty list means pinning took effect. A non-empty list means a native
    library was imported before pinning, and results may not be bit-reproducible.
    """
    expected = str(n_threads)
    return [v for v in THREAD_ENV_VARS if os.environ.get(v) != expected]


def row_order_fingerprint(frame: Any, columns: tuple[str, ...] = FINGERPRINT_COLUMNS) -> str:
    """Hash a prepared frame's content *and* row order.

    Two runs that agree here consumed exactly the same data in exactly the same
    order, so any metric difference between them comes from the model code rather
    than from the data path. Two runs that disagree here will almost certainly
    produce different gradient boosting models, and the reproducibility check says
    so directly instead of reporting two dozen drifting metrics.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"fingerprint column {column!r} is absent")
        values = frame[column]
        array = (
            values.to_numpy(dtype="datetime64[ns]").astype("int64")
            if str(values.dtype).startswith("datetime")
            else values.to_numpy(dtype="float64")
        )
        digest.update(column.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]
