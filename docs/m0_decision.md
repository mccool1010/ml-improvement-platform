# M0 — Locked decisions

Status: accepted, 2026-09-06. Everything below is implemented and verified by a
run against the real dataset. Supporting detail lives in `docs/data.md` and the
three architecture decision records.

---

## 1. Dataset and exact source

**U.S. Small Business Administration national loan guarantee register** (7(a) and
504 programs), the dataset published with Li, Mickel & Taylor (2018), *Should This
Loan be Approved or Denied?*, Journal of Statistics Education 26(1).

- URL: `https://huggingface.co/datasets/MaddRaf/SBAnational/resolve/main/SBAnational.csv`
- SHA-256: `0359128a0b7599e83e4c2e4dcdd781d9121a765237f98d9f0ec1ab3e7c522548`
- 899,164 rows, 27 columns, approvals from 1961-12-07 to 2014-06-25, 179 MB

No authentication is required, so anyone can reproduce the project. The checksum
is verified on every run.

## 2. Why it fits

Assessed against the six stated criteria, with measured evidence:

| Requirement | Evidence |
|---|---|
| Genuine date field | `ApprovalDate`, `DisbursementDate`, `ChgOffDate`, all real calendar dates spanning 53 years |
| Meaningful imbalance | 5.80% positive overall, moving from 3.1% to 19.1% by period |
| Enough samples | 682,428 usable rows after labelling and exposure restriction |
| Improvable baseline | Logistic regression reaches 0.163 average precision on the holdout; gradient boosting on engineered features reaches 0.701 |
| Real drift, not corruption | 60-month default rate rises from 2.50% (2000) to 23.89% (2007) as the financial crisis passes through the book |
| Delayed labels are intrinsic | `ChgOffDate` present for 157,511 of 157,558 charge-offs; median 1,410 days from disbursement to charge-off |

The decisive property is the last one. Delayed ground truth is not simulated here.
The register records when each outcome became knowable, so the delayed-label design
can be built and tested against reality rather than asserted.

Lending Club was the main alternative and was rejected because its delayed-label
problem and its drift period are the same period, so labels for the drifted cohort
do not exist in the file at all. Full reasoning in ADR-001.

## 3. Time-based split strategy

Strictly ordered by approval date. Contiguous, non-overlapping, never shuffled.

| Split | Approval window | Rows | Positives | Rate |
|---|---|---|---|---|
| train | 2000-01-01 to 2003-12-31 | 150,160 | 4,602 | 3.06% |
| validation | 2004-01-01 to 2004-12-31 | 54,285 | 2,587 | 4.77% |
| test | 2005-01-01 to 2005-12-31 | 53,487 | 3,712 | 6.94% |
| production_stream | 2006-01-01 to 2009-06-30 | 121,664 | 23,266 | 19.12% |

`production_stream` is quarantined from model development entirely. It is replayed
as live traffic in the monitoring, drift and retraining milestones.

Training begins in 2000 because `UrbanRural` was still being populated before then,
so earlier rows follow a schema that no longer exists.

## 4. Prediction target

`target = 1` if the loan **charged off within 60 months of disbursement**, else 0.

A row is included only once its full 60-month horizon has elapsed before the
observation cutoff of 2014-06-25. The inclusion rule depends only on the calendar,
never on the outcome, so the label is unbiased.

The naive alternative, "did it eventually charge off", is rejected. Filtering to
resolved loans keeps 85% of FY2006 approvals but only 26% of FY2010, because
defaults resolve in about four years while healthy loans resolve at maturity. That
filter reports a 69% default rate for FY2008, which is an artifact. Detail in
`docs/data.md`.

The horizon captures 68.9% of eventual charge-offs. Later defaults are labelled
negative. This is a stated limitation and the horizon is configurable.

**The population is also restricted to loans with `Term >= 60` months**, so every
loan is at risk for the whole horizon and none matures early. Without this the
default rate cliffs from 41.0% for three-to-five year loans to 8.3% for five-to-
seven year loans, exactly at the horizon boundary, and `Term` alone reproduces 96%
of full-model performance. The model would be learning the shape of the
observation window rather than credit risk. The restriction costs 19.4% of rows.

## 5. Baseline model

Logistic regression on the core register fields, with median imputation, scaling,
and one-hot encoding, all inside one scikit-learn pipeline so preprocessing travels
with the model.

Measured on the 2005 holdout:

| Metric | Train | Validation | Test |
|---|---|---|---|
| Base rate | 0.0306 | 0.0477 | 0.0694 |
| Average precision | 0.1025 | 0.1298 | 0.1632 |
| ROC AUC | 0.7490 | 0.7501 | 0.7258 |
| Brier score | 0.0287 | 0.0441 | 0.0639 |
| Brier skill vs base rate | +0.035 | +0.029 | +0.010 |
| Precision at 10% review | 0.1038 | 0.1555 | 0.2081 |
| Recall at 10% review | 0.3385 | 0.3262 | 0.2998 |
| Lift at 10% review | 3.39x | 3.26x | 3.00x |
| Calibration ratio | 1.00 | 0.65 | 0.46 |

Three findings from building it, all of which changed the design. The first two
were measured before the uniform-exposure restriction was introduced.

- **Balanced class weights were removed.** They left ranking essentially unchanged
  but pushed mean predicted probability to 0.46 against a true rate of 0.19,
  scoring worse on Brier than a constant base-rate predictor. Removing them
  improved average precision, ROC AUC and Brier together.
- **A fixed precision target was abandoned.** The baseline could not reach 50%
  precision at any threshold, so that gate would have been permanently
  uninformative. The operating point is now a 10% review capacity.
- **The population was restricted to uniform exposure.** See section 4. This is the
  correction that turned an apparently strong baseline into an honest one: measured
  average precision on the holdout fell from 0.375 to 0.163 once the model could no
  longer exploit the horizon boundary.

The calibration ratio falling from 1.00 to 0.46 across the splits is the drift
showing up in the baseline itself. The model systematically under-predicts risk as
the crisis approaches, while its ranking ability barely moves. That is exactly the
failure mode a ranking-only quality gate would miss, and it is why calibration is
gate 4 in ADR-003.

## 6. Candidate optimization strategy

Histogram gradient boosting on an engineered feature set, tuned with Optuna at M5.

Engineered features, all computable at decision time: SBA guarantee ratio,
disbursement ratio, log gross approval, amount per employee, jobs supported,
approval-to-disbursement gap, term in years and bucketed, NAICS sector,
same-state-lender flag, franchise flag, and approval month.

Untuned, this already reaches **0.7012 average precision** and **0.9600 ROC AUC** on
the 2005 holdout, against the baseline's 0.1632 and 0.7258. Recall at the 10%
review capacity rises from 0.2998 to 0.7718, and Brier skill from +0.010 to +0.390.
Optuna's job at M5 is to improve on that honestly and to record the trial history,
not to manufacture the headline gain.

Search space: learning rate, tree depth and leaf count, minimum samples per leaf,
L2 regularisation, and iteration count with early stopping on the 2004 validation
split. The 2005 test split is not touched during search.

## 7. Drift signal

Computed on incoming features against the training reference distribution, with no
labels required:

- Population stability index per continuous feature, alert above 0.2
- Chi-squared or total variation distance per categorical feature
- Prediction score distribution shift against the reference scoring distribution
- Flagged-rate shift, meaning the share of applications above the review threshold
- Schema and null-rate drift via the Pandera schema, which catches upstream changes

**Drift may trigger investigation and retraining. It may never, on its own, reject a
model or roll back a deployment.** A change in the mix of applications is not
evidence that the model got worse.

The production stream provides a genuine drift event: input distributions and the
outcome rate both move sharply from 2006 onward, with no synthetic corruption.

## 8. Delayed performance evaluation

Every prediction is logged with its identifier, model version, score, decision
threshold, and `label_available_date`. For an in-horizon default that date is the
charge-off date; otherwise it is the end of the 60-month horizon.

A scheduled evaluation job joins predictions to labels only as they mature, then
scores each model version on the traffic it actually served. Nothing is evaluated
before its label could genuinely exist.

Because a full readout takes five years, the system also reports **partial-horizon
performance**: default-within-12-months and within-24-months, computed on the
subset old enough to support them. These arrive sooner and are directionally
useful, and every report states which horizon it used.

Realised performance produces the long-run improvement record and can trigger a
retraining cycle. It is never on the deployment critical path.

## 9. Promotion gates

A candidate is promoted only if it passes all eight. Any failure rejects it and
leaves production untouched. Full rationale in ADR-003.

1. Data validation passes and no row has an unelapsed label horizon.
2. Average precision exceeds production by at least 0.01 absolute on the same holdout.
3. ROC AUC not more than 0.005 below production.
4. Brier score no worse than production, and positive Brier skill against the base rate.
5. Recall at the 10% review capacity not below production.
6. No segment of at least 1,000 rows loses more than 0.02 average precision, across industry sector, loan size band and region.
7. p99 single-prediction latency within the serving budget.
8. Run record carries a clean git revision, config fingerprint and data checksum.

Candidate and production are always compared on the same window, never against a
historical number from a different period, because the base rate moves too much for
that to mean anything.

## 10. KServe serving architecture

**KServe is the canonical model-serving path.** It owns traffic splitting, canary
rollout, revision management and rollback.

**FastAPI is the application layer, not a second serving path.** It validates
requests, exposes health and readiness, reports model-version metadata, emits
application-tier metrics, and calls the KServe endpoint. It also serves as the
local development server with a locally loaded model.

There is deliberately no second Kubernetes Deployment serving the model directly.
Two production serving paths for one model would drift apart and would make a
canary proven on one meaningless for the other. Full reasoning in ADR-002.

## 11. Canary rollback signals

Operational and real-time only:

- request error rate by status class
- latency at p50, p95, p99
- timeout and saturation rate
- prediction distribution shift between canary and production on the same traffic window
- flagged-rate shift
- readiness and restart behaviour of the served revision

Model accuracy is explicitly **not** a canary signal. With a 60-month label horizon
it cannot be. A canary rollback therefore concludes that the candidate was
operationally unsafe, never that it was less accurate, and the failure report must
say so.

## 12. Measurable success criteria

| # | Criterion | Target | Status |
|---|---|---|---|
| 1 | Reproducible baseline | Same metrics from the same commit, config and data checksum | Met at M1 |
| 2 | Measurable improvement | Candidate beats baseline average precision by at least 0.05 absolute on the 2005 holdout | Met: 0.1632 to 0.7012 |
| 3 | Optimisation is recorded | At least 50 Optuna trials logged with parameters and scores | M5 |
| 4 | Gates reject a bad model | A deliberately degraded candidate is rejected, and production is unchanged | M6, M15 |
| 5 | Versioned and traceable | Every promoted model traces to a commit, config fingerprint and data checksum | M6 |
| 6 | Serving latency | p99 single prediction under 100 ms locally | M7, M8 |
| 7 | Drift is detected | Replaying the 2006+ stream, where the default rate is 19.1% against 6.9% in test, raises drift alerts that the 2005 window does not | M13 |
| 8 | Retraining improves the drifted period | A model retrained through 2006 beats the 2005-era model on 2007 traffic | M13 |
| 9 | Canary rollback works | An injected error-rate or latency breach halts rollout and restores the previous revision | M14 |
| 10 | Delayed evaluation works | Predictions are scored only after `label_available_date`, verified by a test that fails if an unmatured label is used | M13 |
| 11 | Failure scenarios documented | All six scenarios in the plan demonstrated with evidence | M15 |
| 12 | Another developer can reproduce it | Clean clone to trained baseline following the README only | M16 |

---

## What is already built

Milestone M1 is complete and verified against the real dataset:

- `scripts/download_data.py` fetches and checksum-verifies the register
- `scripts/validate_data.py` runs both schemas and reports split composition
- `scripts/train.py` trains either model and writes a JSON run record carrying the
  git revision, config fingerprint, data checksum, library versions, seed, split
  composition and full metrics
- 41 unit tests cover the century-corrected date parsing, the unbiased labelling
  rule, the uniform-exposure restriction, leakage removal, split ordering, and the
  evaluation metrics
- `ruff`, `ruff format` and `mypy --strict` all pass
