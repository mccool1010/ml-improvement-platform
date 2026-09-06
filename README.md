# ML Improvement Platform

A closed-loop ML platform that decides whether a newly trained model is genuinely
better than the one in production, and promotes it only if it is.

The ML task is a vehicle. The engineering is the point: reproducible training,
automated evaluation, quality gates that can reject a candidate, safe promotion,
drift detection, delayed-label evaluation, canary rollout and rollback.

```
experiment -> evaluate -> compare -> promote or reject -> deploy
     ^                                                      |
     |                                                      v
  retrain <- detect degradation <- monitor <----------------+
```

## The problem

Predict whether an SBA-guaranteed small business loan will charge off within 60
months of disbursement, using the U.S. Small Business Administration national loan
guarantee register: 899,164 approvals from 1961 to 2014.

Three properties made this dataset the right choice, and they shape the whole
design:

- **Ground truth is genuinely delayed.** The median gap between disbursement and
  charge-off is 1,410 days. A prediction made today is not verifiable for years.
  The system never pretends otherwise.
- **Drift is real, not injected.** The 60-month default rate rises from 2.50% for
  2000 approvals to 23.89% for 2007 as the financial crisis moves through the book.
  No data is artificially corrupted anywhere in this project.
- **The data is genuinely messy.** Two-digit years spanning 1961 to 2014, currency
  as `"$60,000.00 "`, nominally Y/N columns holding stray codes, and a field that
  was still being populated partway through the period.

Full reasoning in [docs/decisions/ADR-001-model-choice.md](docs/decisions/ADR-001-model-choice.md).

## Results so far

Time-ordered splits, no shuffling. Model development never touches 2006 onward.

| Split | Approval window | Rows | Default rate |
|---|---|---|---|
| train | 2000 to 2003 | 150,158 | 3.04% |
| validation | 2004 | 54,284 | 4.72% |
| test | 2005 | 53,487 | 6.83% |
| production stream | 2006 to mid-2009 | 121,664 | 18.91% |

Measured on the 2005 holdout:

| Metric | Baseline | Candidate |
|---|---|---|
| Average precision | 0.1629 | 0.7036 |
| ROC AUC | 0.7299 | 0.9603 |
| Brier skill vs base rate | +0.012 | +0.395 |
| Recall at 10% review capacity | 0.3070 | 0.7820 |
| Lift at 10% review capacity | 3.07x | 7.82x |

The baseline is logistic regression on core register fields. The candidate is
histogram gradient boosting on engineered features, not yet tuned.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev              # exact dependency set from uv.lock

uv run python -m ml_platform download     # fetch and checksum-verify the register
uv run python -m ml_platform validate     # run both schemas, report split composition
uv run python -m ml_platform train --model baseline
uv run python -m ml_platform train --model candidate
```

To verify the whole thing reproduces the locked results:

```bash
uv run python -m ml_platform reproduce
```

That retrains both models and compares all 48 metrics against
`configs/reference.yaml`, exiting non-zero if anything moves. On the reference
platform every metric is bit-exact. See [docs/reproducibility.md](docs/reproducibility.md).

Each training run writes a JSON record to `artifacts/reports/` carrying the git
revision, config fingerprint, dataset checksum, row-order fingerprint, lockfile
checksum, library versions, seed, thread pinning, split composition and every
metric. A number without that provenance is not evidence.

## Experiment tracking (MLflow)

Every training run is recorded in MLflow. Tracking is on by default and writes to
a local SQLite store; both the database and the artifact directory resolve from
the project root, so the location does not depend on where you run the command.

```bash
uv run python -m ml_platform train --model baseline
uv run python -m ml_platform train --model candidate

# Browse the runs at http://127.0.0.1:5000
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Each run records the model and its hyperparameters, the labelling rules and split
boundaries, every metric already in the JSON run record, and provenance tags for
the commit, config fingerprint, dataset checksum and row-order fingerprint. The
run record and the fitted pipeline are attached as artifacts.

MLflow **records** runs; it does not decide anything. The JSON records under
`artifacts/reports/` and the locked `configs/reference.yaml` stay authoritative,
and `ml_platform reproduce` never reads MLflow. Tracking failures are logged and
swallowed, so a recorder can never fail a training run. Set `tracking.enabled` to
`false` in `configs/base.yaml` to turn it off.

## Hyperparameter optimisation (Optuna)

```bash
uv run python -m ml_platform optimize                 # configured budget
uv run python -m ml_platform optimize --trials 10     # shorter run

uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
# study runs live in the `sba-loan-default-optimization` experiment
```

**Search space**, defined in `configs/model.yaml` so widening a bound shows up in
the config fingerprint. Five parameters, chosen because each is a real lever on
gradient boosting: `learning_rate` (0.01 to 0.3, log), `max_iter` (100 to 400,
step 50), `max_leaf_nodes` (15 to 63), `min_samples_leaf` (20 to 200), and
`l2_regularization` (1e-4 to 10, log).

**Objective:** average precision on the **validation** split, maximised. The test
split is never scored during a search; selecting hyperparameters against it would
turn the final honest number into a training metric, and the objective raises if
configured to use it.

**Budget:** 25 trials, sampled by `TPESampler` with an explicit seed. Optuna seeds
itself from entropy otherwise, so a study would not be reproducible.

**Best candidate:** chosen by Optuna from values the project's own
`evaluate_model` produced. A trial that raises is caught, recorded in `FAIL`
state and excluded from `best_trial`, so a failed trial can never be selected.
MLflow records the decision and never makes it.

The winner is then trained through the ordinary `run_training` path and compared
against a freshly trained baseline, so both sides of the comparison are real
`RunRecord`s from identical machinery. Deciding whether the improvement is large
enough to deploy is M6, not this milestone.

MLflow layout: one parent run per study, one nested run per trial, with the best
parameters and `study.json` on the parent.

Development commands:

```bash
pytest                      # 357 tests; add -m "not slow" to skip full-dataset runs
ruff check src tests scripts
ruff format src tests scripts
mypy                        # strict mode
```

## Design decisions worth knowing

**The label is a fixed 60-month horizon, not "did it eventually default".** The
obvious label is biased: defaults resolve in about four years while healthy loans
resolve at maturity, so filtering to resolved loans keeps far more defaults than
healthy loans and reports an implausible 69% default rate for 2008 approvals.

**The population is restricted to loans of five years or more.** Otherwise the
target partly measures whether the loan even lasted long enough to default, and
`Term` alone reproduces 96% of model performance by learning the horizon boundary.

**Drift and performance degradation are separate signals on separate clocks.**
Input drift is immediate and may trigger retraining, but may never on its own
reject a model. Realised performance takes years and drives the improvement record.
Canary rollback uses only operational signals: error rate, latency, timeouts,
prediction distribution shift and serving health. Never accuracy, because accuracy
does not arrive in time.

**KServe is the canonical serving path.** FastAPI is the application layer in front
of it, not a second way to serve the same model.

**Row order changes the model, so it is pinned and fingerprinted.** Approval dates
tie thousands of times per day, and gradient boosting accumulates in row order.
Changing only the sort algorithm was measured to move candidate average precision
by 0.0017. Every run hashes the prepared data's content and order, and the
reproducibility check compares that hash exactly.

## Documentation

| Document | Contents |
|---|---|
| [docs/m0_decision.md](docs/m0_decision.md) | The locked M0 decisions, all twelve |
| [docs/architecture.md](docs/architecture.md) | Component map and what each technology is for |
| [docs/data.md](docs/data.md) | Source, labelling, splits, drift, data quality, leakage |
| [docs/model_lifecycle.md](docs/model_lifecycle.md) | The loop, and the three clocks |
| [docs/reproducibility.md](docs/reproducibility.md) | What is pinned, the tolerances, and remaining nondeterminism |
| [docs/testing.md](docs/testing.md) | The test tiers and the ML assumptions each one protects |
| [ADR-001](docs/decisions/ADR-001-model-choice.md) | Dataset, label and model family |
| [ADR-002](docs/decisions/ADR-002-serving.md) | Why KServe is the only serving path |
| [ADR-003](docs/decisions/ADR-003-promotion-strategy.md) | Promotion gates, drift, rollback |

## Status

| Milestone | State |
|---|---|
| M0 architecture and problem selection | Complete |
| M1 ML baseline | Complete |
| M2 reproducible training | Complete, 48 of 48 metrics reproduce bit-exactly |
| M3 automated testing | Complete, 357 tests across unit, integration and regression |
| M4 to M16 | Planned |

Built milestone by milestone, each verified by running it.

## Licence

MIT. See [LICENSE](LICENSE).
