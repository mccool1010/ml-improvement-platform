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

## Quality gates and the model registry

```bash
uv run python -m ml_platform promote                    # evaluate and register if it passes
uv run python -m ml_platform promote --no-register      # gates only, dry run
```

Exit code is 0 when the candidate is promoted and 1 when it is rejected, so this
can gate a pipeline.

**Decisions use the validation split.** The test split is held-out evidence; the
gate evaluator raises if asked to decide on it.

**Production is identified explicitly**, never by recency. It is the model
carrying the `production` alias on the registered model
`sba-loan-default-classifier`. If nothing is registered yet, the comparison falls
back to the configured `bootstrap_model`, trained fresh. There is no third path,
so an accidental run can never become the incumbent.

Seven configurable gates, all in `configs/base.yaml`:

| Gate | Default | Why |
|---|---|---|
| `min_improvement` | AP delta >= 0.01 | Below this the difference is not distinguishable from noise |
| `minimum_metric` | AP >= 0.25 | An absolute floor, so beating a poor incumbent is not enough |
| `roc_auc_regression` | within 0.005 | Stops trading general ranking for a narrow gain |
| `calibration` | Brier no worse, skill > 0 | Probabilities must stay honest, not just the ordering |
| `recall_regression` | no decline | The operating point the model actually informs |
| `latency` | mean <= 1.0 ms/prediction | Batch throughput from the evaluation pass, not a single-request p99 |
| `reproducibility` | clean revision + lockfile | An unreproducible result is not evidence |

A candidate that fails any mandatory gate is **not registered**. It keeps its
MLflow run and logged model artifact, but gets no registered version and the
production alias does not move. `register_candidate` refuses a failed report
outright, so a wiring mistake cannot route a rejected model into the registry.

The gate report, not MLflow, decides. MLflow records the outcome.

## Inference API

```bash
uv run python -m ml_platform serve                      # http://127.0.0.1:8000
# interactive docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness. Answers whenever the process is up |
| `GET /ready` | Readiness. Whether a promoted model is loaded, and why not if it is not |
| `GET /model` | Which model is serving, and where it came from |
| `POST /predict` | Score one or more loan applications |

Example:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "applications": [{
    "term_months": 84, "employees": 12, "jobs_created": 3, "jobs_retained": 8,
    "gross_approved": 250000, "sba_approved": 187500, "disbursed": 250000,
    "state": "CA", "bank_state": "CA", "revolving_line_of_credit": "N",
    "low_doc": "N", "urban_rural": 1, "new_business": 1, "naics": "722410",
    "franchise_code": 0, "approval_date": "2005-06-15",
    "disbursement_date": "2005-07-20"
  }]
}'
```

The model is resolved through the `production` registry alias, the same mechanism
M6 promotion uses, and loaded **once at startup**. There is no fallback to the
newest run or to a file on disk: with no promoted model the service starts,
`/health` answers, and `/ready` reports why it cannot serve. Serving something
nobody promoted would defeat the promotion system.

Requests are loan applications in the lender's own terms. Features are derived by
the same `features.engineering` code the model was trained with, so the served
representation cannot drift from the trained one. Every response carries the
model name, version and both run identities, so a score is traceable.

## Container

```bash
docker build -f docker/Dockerfile -t ml-platform-api:latest .

# The MLflow store is mounted, not baked in, so promoting a model needs no rebuild.
# The artifact mount target is the absolute path MLflow recorded in the database,
# not a tidy one; see the note below. `--mount` is required because `-v` cannot
# parse a target containing a colon.
docker run --rm -p 8000:8000 \
  --mount "type=bind,source=$PWD/mlflow.db,target=/app/mlflow.db" \
  --mount "type=bind,source=$PWD/mlartifacts,target=/$PWD/mlartifacts" \
  ml-platform-api:latest

curl -fsS localhost:8000/health
curl -fsS localhost:8000/ready
```

With no store mounted at all the container still starts and serves `/health`; it
reports `/ready` as `not_ready` with the reason. That is the intended behaviour,
and it is what CI asserts.

Two stages. The builder resolves dependencies with `uv sync --frozen` from the
committed lockfile, so the image gets the exact versions the reproducibility
contract records. The runtime stage carries only the virtualenv, `src/` and
`configs/`, and runs as a non-root user on port 8000.

The image deliberately excludes the 179 MB register (the API scores, it does not
train) and the MLflow store. Baking the store in would freeze one registry state
into the image, so a promotion would mean a rebuild.

Native thread pools are pinned in the image environment, because uvicorn imports
the app directly and never passes through the CLI bootstrap that pins them.

The `HEALTHCHECK` uses `/health`, not `/ready`. A container holding no promoted
model is alive and should not be restarted; withholding traffic is what `/ready`
is for.

One real limitation, found by running this. MLflow records each experiment's
artifact root in the tracking database as an absolute host URI, here
`file:///C:/mlops/mlartifacts`. That is data, not configuration, so a container
that mounts the artifacts anywhere else resolves them to nothing and reports
`No such artifact: ''`. Mounting at the recorded path is a workaround for one
machine. The real fix is a tracking server with a shared artifact store, which
belongs to a later milestone and is not hacked around here.

Development commands:

```bash
pytest                      # add -m "not slow" to skip full-dataset runs
ruff check .
ruff format --check .
mypy                        # strict mode
```

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and every pull request against it, in four independent jobs: static checks
(`ruff`, `ruff format --check`, `mypy --strict`), the test suite, a real
reproduction of the locked reference metrics on the genuine register, and a build
and smoke-test of the container image. Dependencies install from `uv.lock` with
`--frozen`, the runner image and `uv` version are pinned, and the workflow holds
no secret and pushes nothing.

CI verifies the container starts and correctly reports itself **not ready** with
no registry mounted; it cannot yet load a real promoted model, because the local
MLflow store records absolute host paths. The reasoning, the local equivalents of
every CI command, and that limitation in full are in [docs/ci.md](docs/ci.md).

## Kubernetes

```bash
docker build -f docker/Dockerfile -t ml-platform-api:m10 .
kubectl apply -k k8s/base
kubectl -n ml-platform rollout status deploy/mlflow

# The registry starts empty, so the API pods run but stay out of rotation.
kubectl -n ml-platform port-forward svc/mlflow 5000:5000 &
uv run python scripts/seed_registry.py --destination http://localhost:5000

kubectl -n ml-platform port-forward svc/inference-api 8080:80 &
curl -s localhost:8080/ready
```

Two workloads: the inference API, and an MLflow tracking server started with
`--serve-artifacts`. The server is what makes the deployment possible at all.
M8 established that the local store records an absolute host artifact path in its
database, which no pod can resolve; a server that brokers artifacts over HTTP
records `mlflow-artifacts:/...` instead, so nothing outside the cluster is ever
named. It runs the project's own image, so the MLflow build that writes the
registry is the one from `uv.lock` that reads it.

The API finds it through cluster DNS at `http://mlflow:5000`, supplied by a
ConfigMap. That is the only value that comes from Kubernetes, because it is the
only one that differs between a laptop and a cluster. There is no Secret, because
there is no credential yet.

Liveness asks `/health`, readiness asks `/ready`, and a pod with no promoted
model stays running and out of the Service's endpoints rather than restarting in
a loop. See [docs/kubernetes.md](docs/kubernetes.md) for the full runbook and the
remaining limitations.

## KServe

The model is served by a KServe `InferenceService`, not by the FastAPI
Deployment. FastAPI is the application tier in front of it: it validates the
application, resolves the `production` alias in MLflow, reports which version
answered, and calls the model tier for the score. One production serving path,
as [ADR-002](docs/decisions/ADR-002-serving.md) decided at M0.

```bash
kubectl apply -k k8s/kserve
kubectl -n ml-platform get isvc sba-loan-default
```

KServe v0.20.0 in RawDeployment mode, with cert-manager and no Knative, Istio or
Gateway API. The predictor runs the project's own image because the registered
artifact pins `scikit-learn==1.9.0` and is cloudpickle-serialised, which the
stock `seldonio/mlserver` runtime cannot satisfy. It reads the artifact from the
tracking server's volume at the path the registry records for the promoted
version. See [docs/kserve.md](docs/kserve.md).

## Observability

```bash
kubectl apply -k k8s/monitoring
kubectl -n ml-platform port-forward svc/grafana 3000:3000 &
```

Both serving tiers expose Prometheus metrics and export OpenTelemetry traces.
Grafana opens onto a provisioned dashboard with Prometheus already wired in, so
a fresh cluster needs no clicking. Jaeger receives OTLP directly -- there is no
collector in between, because with one producer and one backend it would add a
hop and a failure mode and nothing else.

A `/predict` trace crosses both tiers in one trace, because the outgoing HTTP
client and the model tier's server are both instrumented and `traceparent`
survives the hop.

The metrics are the operational signals [ADR-003](docs/decisions/ADR-003-promotion-strategy.md)
draws rollback authority from: request rate, errors, latency, upstream timeouts
and serving health. Accuracy is deliberately absent -- it arrives years late and
belongs to a different clock. Every label is drawn from a fixed set, and the
route label is a template rather than a path, so a thousand distinct URLs are one
series and not a thousand. See [docs/observability.md](docs/observability.md).

## Drift and retraining

```bash
python -m ml_platform drift                          # compare training vs production window
python -m ml_platform drift --scenario lending_shift # controlled demonstration
python -m ml_platform retrain                        # retrain if drifted, then run the gates
```

Drift is measured on the engineered features, by PSI with KS and chi-square as
corroboration, against a deterministic slice of the quarantined 2006-mid-2009
production stream. It reads **inputs only**: it can trigger a retraining attempt
and can never reject a model, because saying a model got worse needs matured
labels that take five years to arrive on this dataset.

Retraining adds the current window to the training data and produces a
*candidate*. That candidate goes through the existing M6 gates on the same
validation split as any other, so a retrained model that is worse is rejected and
production is untouched. Run for real, the retrained candidate improved
validation AP by 0.0024 and was **rejected** for being inside the noise band the
gate exists to filter.

Everything carries a `drift_event_id`, so a promotion can be traced back to the
check that caused it. See [docs/drift.md](docs/drift.md), which also states
exactly which labels retraining is allowed to use and why.

## Canary and rollback

A gated candidate is registered under a **canary** alias, not the production one,
and the application tier routes a configured share of traffic to it. Only after
the canary passes does the production alias move.

KServe runs in RawDeployment mode, which has no traffic-splitting primitive, so
the split happens in the tier that already fronts the model. Routing is
deterministic and sticky -- a SHA-256 of the routing key into 10,000 buckets --
so the same application always reaches the same tier and a test can assert an
exact split. No service mesh was introduced.

Rollback signals are operational only: error rate, latency, upstream timeouts and
serving health, each compared against both an absolute ceiling and the incumbent.
Accuracy is deliberately not among them, and a test asserts it. A failing canary
never falls back to production, because that would make the error rate the
decision rests on read as zero.

Demonstrated end to end against a real MLflow registry: a healthy candidate
passing all seven checks and moving production to v2; an unreachable candidate
producing 92 real 503s and leaving production at v1; and a healthy tier with a
19.6% error rate also leaving production at v1. See [docs/canary.md](docs/canary.md).

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
| [docs/ci.md](docs/ci.md) | What CI checks, how to run those checks locally, and what it cannot test yet |
| [docs/kubernetes.md](docs/kubernetes.md) | The cluster architecture, the tracking server it needed, and the deploy runbook |
| [docs/kserve.md](docs/kserve.md) | The two serving tiers, the runtime choice, and the KServe install |
| [docs/observability.md](docs/observability.md) | Metrics, the dashboard, tracing, and what is deliberately not measured |
| [docs/drift.md](docs/drift.md) | Drift methodology, the controlled scenario, and which labels retraining may use |
| [docs/canary.md](docs/canary.md) | Traffic splitting, the rollback signals, and why the alias moves last |
| [ADR-001](docs/decisions/ADR-001-model-choice.md) | Dataset, label and model family |
| [ADR-002](docs/decisions/ADR-002-serving.md) | Why KServe is the only serving path |
| [ADR-003](docs/decisions/ADR-003-promotion-strategy.md) | Promotion gates, drift, rollback |

## Status

| Milestone | State |
|---|---|
| M0 architecture and problem selection | Complete |
| M1 ML baseline | Complete |
| M2 reproducible training | Complete, 48 of 48 metrics reproduce bit-exactly |
| M3 automated testing | Complete, unit, integration and regression tiers |
| M4 experiment tracking | Complete, MLflow records runs; the JSON records stay authoritative |
| M5 model optimization | Complete, Optuna study with nested runs |
| M6 quality gates and registry | Complete, gates decide on validation, never on the test split |
| M7 inference API | Complete, FastAPI resolving the `production` registry alias |
| M8 containerisation | Complete, two-stage image, non-root, no data or store baked in |
| M9 CI/CD | Complete, GitHub Actions: static checks, tests, reproduction, image build |
| M10 Kubernetes | Complete, API and an MLflow tracking server, with artifacts served over HTTP |
| M11 KServe serving | Complete, KServe owns the model tier; FastAPI is the application tier |
| M12 observability | Complete, Prometheus, Grafana, OpenTelemetry and Jaeger across both tiers |
| M13 drift and retraining | Complete, drift triggers retraining; the M6 gates still decide |
| M14 canary and rollback | Complete, app-tier traffic split; the alias moves only after the canary passes |
| M15 to M16 | Planned |

Built milestone by milestone, each verified by running it.

## Licence

MIT. See [LICENSE](LICENSE).
