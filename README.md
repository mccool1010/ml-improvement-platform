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
| train | 2000 to 2003 | 150,160 | 3.06% |
| validation | 2004 | 54,285 | 4.77% |
| test | 2005 | 53,487 | 6.94% |
| production stream | 2006 to mid-2009 | 121,664 | 19.12% |

Measured on the 2005 holdout:

| Metric | Baseline | Candidate |
|---|---|---|
| Average precision | 0.1632 | 0.7012 |
| ROC AUC | 0.7258 | 0.9600 |
| Brier skill vs base rate | +0.010 | +0.390 |
| Recall at 10% review capacity | 0.2998 | 0.7718 |
| Lift at 10% review capacity | 3.00x | 7.72x |

The baseline is logistic regression on core register fields. The candidate is
histogram gradient boosting on engineered features, not yet tuned.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

python scripts/download_data.py     # fetch and checksum-verify the register
python scripts/validate_data.py     # run both schemas, report split composition
python scripts/train.py --model baseline
python scripts/train.py --model candidate
```

Each training run writes a JSON record to `artifacts/reports/` carrying the git
revision, config fingerprint, data checksum, library versions, seed, split
composition and every metric. A number without that provenance is not evidence.

Development commands:

```bash
pytest                      # unit tests
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

## Documentation

| Document | Contents |
|---|---|
| [docs/m0_decision.md](docs/m0_decision.md) | The locked M0 decisions, all twelve |
| [docs/architecture.md](docs/architecture.md) | Component map and what each technology is for |
| [docs/data.md](docs/data.md) | Source, labelling, splits, drift, data quality, leakage |
| [docs/model_lifecycle.md](docs/model_lifecycle.md) | The loop, and the three clocks |
| [ADR-001](docs/decisions/ADR-001-model-choice.md) | Dataset, label and model family |
| [ADR-002](docs/decisions/ADR-002-serving.md) | Why KServe is the only serving path |
| [ADR-003](docs/decisions/ADR-003-promotion-strategy.md) | Promotion gates, drift, rollback |

## Status

| Milestone | State |
|---|---|
| M0 architecture and problem selection | Complete |
| M1 ML baseline | Complete |
| M2 reproducible training | Partly done, provenance and config layering in place |
| M3 automated testing | Unit tests for the data and evaluation layers |
| M4 to M16 | Planned |

Built milestone by milestone, each verified by running it.

## Licence

MIT. See [LICENSE](LICENSE).
