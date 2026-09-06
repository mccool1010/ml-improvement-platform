# Model lifecycle

The loop this platform exists to operate:

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    v                                              │
  data ──> validate ──> prepare ──> split ──> train ──> evaluate ──┤
                                                            │      │
                                                            v      │
                                              compare vs production│
                                                            │      │
                                                    ┌───────┴──────┴──┐
                                                    │  quality gates  │
                                                    └───┬─────────┬───┘
                                                 pass   │         │  fail
                                                        v         v
                                                   registry   rejected,
                                                        │     production
                                                        v     unchanged
                                                  canary rollout
                                                        │
                                            ┌───────────┴───────────┐
                                            │ operational signals   │
                                            └───┬───────────────┬───┘
                                        healthy │               │ breach
                                                v               v
                                            promote          rollback
                                                │
                                                v
                                          serve + monitor
                                                │
                             ┌──────────────────┴──────────────────┐
                             v                                     v
                   input drift (immediate)          realised performance (delayed)
                             │                                     │
                             └──────────────┬──────────────────────┘
                                            v
                                       retraining
```

## The three clocks

This is the design decision that shapes everything else. A prediction made today
gets its label in five years, so signals are separated by how fast they arrive and
by what each is allowed to decide.

| Signal | Latency | May trigger | May not |
|---|---|---|---|
| Input drift | Seconds | Investigation, retraining | Reject a model, roll back a deployment |
| Operational health | Seconds to minutes | Canary rollback, alerting | Claim a model is less accurate |
| Realised performance | Months to years | Retraining, the improvement record | Block a deployment in progress |

Collapsing these into one "model health" number would be the single most damaging
simplification available, so the code keeps them separate all the way through.

## Stages

**Validate.** Pandera schemas run before anything is fitted. A raw schema catches a
changed upstream file. A prepared schema catches parsing regressions such as the
two-digit-year century bug. A separate invariant asserts that no retained row has a
label horizon ending after the observation cutoff, which is what keeps the delayed
label design honest.

**Prepare and split.** Cleaning, fixed-horizon labelling, then contiguous
time-ordered windows. The production stream is quarantined from development.

**Train.** One scikit-learn pipeline holds imputation, encoding and the estimator,
so preprocessing is fitted on training data only and travels with the model. This
removes the most common source of train/serve skew.

**Evaluate.** Average precision is primary. ROC AUC, Brier score and Brier skill,
and precision and recall at the 10% review capacity are reported alongside. Lift
over base rate accompanies every figure because the base rate moves from 6.8% to
33.8% across the study period, so raw average precision is not comparable across
periods.

**Compare and gate.** Candidate against current production, on the same window,
against the eight gates in ADR-003. Every gate records its measured value whether
it passed or failed. A failure leaves production untouched.

**Register.** Passing candidates enter the MLflow registry with their run record:
git revision, config fingerprint, data checksum, library versions, seed, split
composition and full metrics.

**Deploy.** KServe serves the model. FastAPI is the application layer in front of
it, not a second serving path. See ADR-002.

**Canary.** Traffic shifts gradually. Judgement is on operational signals only.

**Monitor.** Application and model metrics to Prometheus, traces through
OpenTelemetry. Every prediction is logged with its model version, score, threshold
and the date its label becomes knowable.

**Detect.** Input drift immediately. Realised performance as labels mature.

**Retrain.** Triggered by drift or by degraded realised performance. Produces a
candidate, which re-enters at the gates. A retrained model is never promoted
automatically without passing them.

## Run records

Every training run writes one JSON file to `artifacts/reports`. That file is the
unit of evidence for the whole project, and later milestones compare candidate
records against the production record to decide promotion. A run from a dirty
working tree is recorded as dirty, because its numbers cannot be reproduced from
the commit alone.
