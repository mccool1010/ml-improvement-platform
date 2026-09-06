# ADR-003: Promotion gates, drift, and rollback signals

- **Status:** Accepted
- **Date:** 2026-09-06
- **Milestone:** M0, implemented across M6, M13 and M14

## Context

A 60-month label horizon means the truth about a prediction made today arrives in
2031. Any design that assumes prompt labels is fiction, and a rollback rule that
waits for accuracy would never fire in time to protect anything.

This ADR fixes which signals are available on which clock, and which decisions
each signal is permitted to make.

## Decision: three signals on three clocks

**1. Input drift. Available immediately, may trigger investigation and retraining,
may never by itself reject or roll back a model.**

Drift is computed on incoming features against the training reference
distribution: population stability index for continuous features, and a
chi-squared or total-variation comparison for categorical ones. Crossing the
threshold raises a drift event and may start a retraining run. It does not say the
model got worse, because it cannot. A shift in the mix of applications is not
evidence of a worse model, and treating it as such would retrain the system into
chasing noise.

**2. Operational and prediction-behaviour signals. Available in seconds to minutes,
and these alone govern canary rollback.**

- request error rate, by status class
- latency, at p50, p95 and p99
- timeout and saturation rate
- prediction distribution shift between canary and production, compared on the
  same traffic window
- flagged-rate shift, meaning the share of applications scored above the decision
  threshold
- readiness and restart behaviour of the served revision

A canary is judged on these and nothing else. They are the only signals that move
fast enough to matter inside a rollout window.

**3. Realised model performance. Available in months to years, governs the
retrospective verdict and the long-run improvement record.**

Every prediction is logged with its identifier, model version, score, and the date
its label becomes knowable. A scheduled job joins predictions to labels as they
mature and evaluates each model version on the traffic it actually served. This is
what proves the improvement claim, and it is deliberately not on the deployment
critical path.

## Consequence: what a canary rollback may and may not conclude

A canary rollback says the candidate was operationally unsafe. It never says the
candidate was less accurate, because that is not knowable in the rollout window.
The failure report must state this distinction, and a rolled-back candidate is
recorded as operationally rejected rather than as a worse model.

## Promotion gates

A candidate is promoted only if it passes every gate. Any failure rejects it and
leaves production untouched.

| # | Gate | Threshold | Rationale |
|---|------|-----------|-----------|
| 1 | Data validation | Prepared schema passes, no rows with an unelapsed label horizon | A candidate trained on invalid data is not evaluable |
| 2 | Improvement | Candidate average precision on the holdout exceeds production by at least 0.01 absolute | Below this the difference is not distinguishable from noise |
| 3 | No ranking regression | Candidate ROC AUC not more than 0.005 below production | Guards against trading general ranking for a narrow gain |
| 4 | Calibration | Candidate Brier score no worse than production, and better than the base-rate-constant predictor | A model whose probabilities are wrong is unusable for a threshold decision |
| 5 | Operating point | Recall at the fixed review capacity not below production | The gate matches the decision the model informs |
| 6 | Subgroup safety | No decline greater than 0.02 average precision on any segment with at least 1,000 rows, across industry sector, loan size band, and region | Stops an aggregate gain that hides a localised regression |
| 7 | Latency | p99 single-prediction latency within the serving budget | An accurate model that misses its latency budget cannot serve |
| 8 | Reproducibility | Run record carries a clean git revision, config fingerprint and data checksum | An unreproducible result is not evidence |

Thresholds live in configuration, not code, and the gate evaluation records every
gate's measured value whether it passed or failed.

## Why an absolute improvement margin

The base rate moves from 6.8% to 33.8% across this dataset, so average precision is
not comparable across periods. Gates 2 and 3 therefore compare candidate against
production **on the same holdout window**, never against a historical number from a
different period. Lift over base rate is recorded alongside every metric so that
cross-period comparisons remain interpretable.

## Rollback

- **Pre-promotion rejection.** Gate failure. Production is untouched and the
  candidate is archived with its gate report.
- **Canary rollback.** An operational signal breaches its threshold during rollout.
  Traffic returns to the previous revision, which stays warm throughout.
- **Post-promotion retrospective rejection.** Matured labels show the promoted
  model underperforming its predecessor on served traffic. This triggers a
  retraining cycle and a recorded finding, not an automatic rollback, because by
  then the world has usually moved on from both models.
