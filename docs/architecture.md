# Architecture

## What this system is for

It decides whether a newly trained model is genuinely better than the one in
production, and promotes it only if it is. Everything else exists to make that
decision trustworthy and repeatable.

The ML task is a vehicle: predicting whether an SBA-guaranteed loan will charge off
within 60 months of disbursement. The engineering is the point.

## Component map

```
  ┌──────────────────────────────────────────────┐
  │ ml_platform.cli        one entry point       │
  │   download validate train reproduce          │
  │ ml_platform.paths      root-anchored paths   │
  │ ml_platform.determinism seeds, threads, sort │
  │ ml_platform.config     layered + fingerprint │
  └──────┬───────────────────────────────────────┘
         v
  ┌─────────────┐
  │ SBA register│  checksum-verified download
  └──────┬──────┘
         v
  ┌─────────────────────────────────────────────┐
  │ ml_platform.data                            │
  │   ingestion   acquire + verify              │
  │   validation  Pandera raw + prepared schemas│
  │   preprocessing  clean, fixed-horizon label │
  │   splitting   time-ordered windows          │
  └──────┬──────────────────────────────────────┘
         v
  ┌──────────────────────┐   ┌────────────────────────┐
  │ ml_platform.features │   │ ml_platform.models      │
  │   core / engineered  │──>│   baseline  pipelines   │
  └──────────────────────┘   │   train     fitting     │
                             │   evaluate  metrics     │
                             └───────┬─────────────────┘
                                     v
                        ┌────────────────────────────┐
                        │ ml_platform.pipelines      │
                        │   train_pipeline           │
                        │   -> JSON run record       │
                        │   reproduce                │
                        │   -> vs configs/reference  │
                        └───────┬────────────────────┘
                                v
                        ┌────────────────────────────┐
                        │ ml_platform.promotion      │  M6
                        │   compare, gates, rollback │
                        └───────┬────────────────────┘
                                v
                        ┌────────────────────────────┐
                        │ MLflow registry            │  M4, M6
                        └───────┬────────────────────┘
                                v
        ┌───────────────────────────────────────────────┐
        │ Serving                                       │
        │   FastAPI  application layer, validation,     │  M7
        │            health, model metadata, metrics    │
        │   KServe   canonical model serving, canary    │  M11
        └───────┬───────────────────────────────────────┘
                v
        ┌────────────────────────────┐
        │ ml_platform.monitoring     │  M12, M13
        │   drift      immediate     │
        │   performance  delayed     │
        │   metrics    Prometheus    │
        └────────────────────────────┘
```

## Design commitments

**One pipeline object holds preprocessing and the estimator.** Imputation and
encoding are fitted on training data only and are serialised with the model. This
removes the most common train/serve skew, where preprocessing is fitted somewhere
the served model cannot reach.

**Splits are time-ordered and the production window is quarantined.** Random
splitting would let 2008 approvals train a model evaluated on 2004. The 2006-2009
window is never seen during model development, so drift experiments are run against
data the model genuinely has not met.

**Signals are separated by latency and authority.** Input drift, operational
health, and realised performance arrive on different clocks and are permitted to
make different decisions. See `docs/model_lifecycle.md`.

**Configuration is layered and fingerprinted.** `base.yaml` plus `model.yaml` plus
an environment overlay, hashed into every run record.

**Every run is self-describing.** The run record carries the git revision and
whether the tree was dirty, the config fingerprint, the dataset checksum, the
row-order fingerprint, the lockfile checksum, library versions, the seed, the
determinism settings, split composition and all metrics. A number without those
is not evidence.

**Results are locked and checked, not asserted.** `configs/reference.yaml` holds
the M1 metrics, split composition and row-order fingerprint. `python -m
ml_platform reproduce` retrains and compares against them, exiting non-zero on any
deviation. See `docs/reproducibility.md`.

**Nothing depends on the working directory.** Every path resolves from the project
root, and no absolute path is written into a run record.

**Validation precedes training.** A schema failure stops the pipeline before a
model is fitted, so a data problem cannot become a quietly degraded model.

## Technology, and what each one is for

Nothing is here to look impressive. Each entry names the requirement it satisfies.

| Technology | Requirement it solves | Milestone |
|---|---|---|
| scikit-learn | Baseline and candidate tabular models in one serialisable pipeline | M1 |
| Pandera | Stop bad data before training, catch upstream schema changes | M1 |
| uv | Reproducible environment, pinned Python 3.12 | M1 |
| pytest | Prove the correctness-critical labelling and splitting logic | M3 |
| MLflow | Compare experiments, version models, record promotion history | M4, M6 |
| Optuna | Systematic hyperparameter search with a recorded trial history | M5 |
| FastAPI + Pydantic | Domain-shaped request validation, health, model metadata | M7 |
| Docker | Reproducible execution of training and serving | M8 |
| GitHub Actions | Automated tests, type checks, security scans, evaluation gates | M9 |
| Kubernetes | Production-style orchestration and health management | M10 |
| KServe | Canonical model serving, traffic splitting, canary, rollback | M11 |
| Prometheus + Grafana | Request rate, latency, errors, prediction behaviour by model version | M12 |
| OpenTelemetry + Jaeger | Trace a prediction across the application and model tiers | M12 |
| Ruff + mypy | Lint and type checking in CI | M1, M9 |
| Bandit + pip-audit | Dependency and code security scanning | M9 |

A component that cannot be tied to a requirement here does not get added.

## Current state

M0, M1 and M2 are complete. The data layer, feature layer, baseline model,
evaluation metrics, run-record provenance and the reproducibility check all run
against the real dataset, with 48 of 48 metrics reproducing bit-exactly from a
fresh checkout. Everything from M3 onward is planned but not built.
