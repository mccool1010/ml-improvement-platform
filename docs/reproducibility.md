# Reproducibility

M2 makes the training path reproducible by construction and, more importantly,
makes any loss of reproducibility a loud failure rather than a drifting number.

## The single command

From a fresh checkout:

```bash
uv sync --frozen --extra dev          # exact dependency set from uv.lock
uv run python -m ml_platform download # fetch and checksum-verify the register
uv run python -m ml_platform reproduce
```

`reproduce` retrains the baseline and the candidate, compares all 48 metrics
against the locked reference in `configs/reference.yaml`, and exits non-zero if
anything falls outside tolerance. It writes a report to
`artifacts/benchmarks/reproducibility-<run-id>.json`.

Everything else is a subcommand of the same entry point: `download`, `validate`,
`train`, `reproduce`. The scripts under `scripts/` are thin wrappers over it, kept
because the project layout calls for them.

## What is pinned, and why

| Control | Mechanism | Why it matters |
|---|---|---|
| Dependency versions | `uv.lock`, installed with `uv sync --frozen`; its SHA-256 is recorded per run | A patch release of scikit-learn can move a split decision |
| Python version | `requires-python = ">=3.12,<3.13"`, interpreter recorded per run | Floating-point and library behaviour vary across versions |
| Dataset bytes | SHA-256 verified on download and recorded per run | A changed upstream mirror would otherwise look like a modelling regression |
| Random seeds | One `seed` in config, applied to `random`, numpy and `PYTHONHASHSEED` | Any sampling or estimator randomness |
| Native thread count | `determinism.n_threads`, pinned across five backend variables before numpy imports | Parallel reductions sum in completion order |
| Row order | `ROW_SORT_KIND`, plus a fingerprint locked in the reference | Gradient boosting bins and accumulates in row order |
| Paths | Everything resolves from the project root, never the working directory | A run must not depend on where it was launched from |
| Configuration | Layered YAML, hashed to a fingerprint recorded per run | A silently edited threshold would otherwise be invisible |

## Row order is the interesting one

Approval dates repeat thousands of times per day, so most rows tie on the sort
key and the sort algorithm decides their relative order. That is not cosmetic.
Measured directly on this dataset, changing only the sort algorithm moved the
candidate model:

| Sort | Average precision | ROC AUC | Recall at 10% |
|---|---|---|---|
| `quicksort` | 0.701151 | 0.959988 | 0.771821 |
| `stable` | 0.702866 | 0.960518 | 0.776131 |

The baseline was unaffected, because logistic regression converges to the same
optimum regardless of row order. Gradient boosting does not: it bins features and
accumulates histograms in row order, so tie order changes split decisions.

`quicksort` is retained because it is the ordering under which the M1 reference
was measured, and it was verified to be deterministic for identical input. But
"deterministic in practice" is a weaker guarantee than "pinned by design": numpy
could change its introsort and silently reorder ties.

That gap is closed by fingerprinting rather than by changing the ordering. Every
run hashes the prepared dataset's content **and** row order into
`dataset.row_order_sha256`, and `reproduce` compares it exactly. If ordering ever
changes, the check reports one clear line about row order instead of two dozen
drifting metrics.

## Tolerances

Two profiles, both defined in `configs/reference.yaml`:

- **`strict`**, default, `1e-6` on every metric. This is the same-platform bound.
  It is not absorbing expected noise: training on this machine is bit-exact. It
  exists only because run records round metrics to six decimal places.
- **`portable`**, `5e-4` on ranking metrics and `2e-3` on threshold-derived ones.
  For a rerun on different hardware, where a different BLAS build may sum in a
  different order. Threshold metrics get more room because they move in steps as
  ranks swap around the cut.

Split composition, the dataset checksum and the row-order fingerprint are compared
**exactly**. A difference in any of those is a fault, not floating-point noise.

## Measured determinism

Verified on the reference platform, CPython 3.12.14 on Windows AMD64:

| Test | Result |
|---|---|
| Two trainings in one process | Byte-identical predictions |
| Separate processes | Byte-identical predictions |
| 1, 4 and 8 native threads | Byte-identical predictions |
| Full `reproduce` against the M1 reference | 48 of 48 metrics bit-exact |

The thread-count result is stronger than expected. scikit-learn's histogram
gradient boosting uses deterministic reductions, so its output did not vary with
thread count here. Thread pinning is kept anyway, because that property is not
guaranteed by the API and a different BLAS on another platform may not share it.

## Remaining nondeterminism

There is none within a platform. The known limits are:

1. **Cross-platform floating point.** A different BLAS build may sum in a
   different order. This is why the `portable` profile exists. Measured once, in CI
   on Ubuntu 24.04: all 48 metrics matched the Windows reference bit-exactly, so
   the bound was not needed there. Other BLAS builds remain unmeasured.
2. **`PYTHONHASHSEED` is set inside the process.** CPython reads it at startup, so
   setting it during a run cannot affect the current interpreter's hashing. No
   part of the pipeline depends on set or dict iteration order today, so this is
   recorded rather than enforced. Export it before launch to remove the caveat.
3. **Row-order ties depend on the sort implementation, not on the data.** Pinned
   and fingerprinted rather than made canonical, because making it canonical would
   change the locked M1 reference. Sorting on approval date plus loan number would
   make order a property of the data alone; that is a deliberate follow-up, not an
   oversight.
4. **Wall-clock fields.** `run_id`, `started_at` and `inference_seconds` differ
   between runs by design and are excluded from comparison.

## Provenance recorded per run

Every run record in `artifacts/reports/` carries:

```
git_revision, git_dirty, config_fingerprint, dataset_sha256, row_order_sha256,
lockfile_sha256, seed, determinism{n_threads, sort_kind, thread_env,
python_hash_seed}, interpreter, platform, libraries{...}
```

`RunContext.is_reproducible()` is true only when the revision is known, the tree
is clean and a lockfile is present. A run from a dirty tree is recorded as dirty
rather than quietly presented as evidence.

No absolute filesystem path is written into a record. Paths are stored relative to
the project root so two machines produce comparable records.
