# Evidence

Claims made elsewhere in this repository, with the observations behind them.

Every number here was produced by running the system and read back out of the
store that recorded it — MLflow runs and tags, benchmark JSON in
`artifacts/benchmarks/`, or a live `kubectl`/HTTP probe. Where something has not
been measured, this page says so rather than estimating it.

Two conventions. **Validation is the decision split**; the test split appears
only where it is explicitly labelled, because a promotion justified by held-out
evidence destroys that evidence. And **a run recorded on a dirty working tree is
marked as such**, because the reproducibility gate treats it as a failure and so
should a reader.

---

## 1. The production model

`sba-loan-default-classifier` v1, alias `production`.

| | |
|---|---|
| MLflow run | `eb35c645f82b446cb49d218d442fe5dc` |
| Platform run | `candidate-20260906T175308Z` |
| Estimator | `HistGradientBoostingClassifier` |
| Feature set | `engineered` (119 features) |
| Promoted over | `logistic_regression_baseline` |
| Quality gates | 7 of 7 passed |
| Git revision | `9ce91520b39840ea83e7b5c77a08572d188dd867` |
| Dataset SHA-256 | `0359128a0b7599e83e4c2e4dcdd781d9121a765237f98d9f0ec1ab3e7c522548` |
| Label | `default_within_60m`, 60-month horizon |

### Validation metrics — the numbers the promotion was decided on

| Metric | Value |
|---|---|
| Average precision | 0.732031 |
| ROC AUC | 0.973004 |
| Recall @ 10% review capacity | 0.868513 |
| Precision @ 10% review capacity | 0.410020 |
| Brier score | 0.022446 |
| Brier skill score | 0.501043 |
| Lift @ capacity | 8.684175 |
| Decision threshold | 0.065487 |

### Test metrics — held out, not used for any decision

| Metric | Value |
|---|---|
| Average precision | 0.703605 |
| ROC AUC | 0.960345 |
| Recall @ 10% review capacity | 0.781978 |
| Brier skill score | 0.394863 |

The drop from validation to test is real and is the honest number: 0.732 → 0.704
average precision. Validation is 2004 and test is 2005, so the gap is a year of
distribution movement, not overfitting to a random split.

### Baseline, for scale

`logistic_regression_baseline` reached validation average precision **0.131498**
against the candidate's 0.732031. The base rate on validation is 4.72%, so the
baseline is barely above chance in ranking terms and the candidate is at roughly
15.5× base rate lift.

### Splits

| Split | Rows | Positives | Positive rate |
|---|---|---|---|
| Train (2000–2003) | 150,158 | 4,561 | 3.04% |
| Validation (2004) | 54,284 | 2,563 | 4.72% |
| Test (2005) | 53,487 | 3,651 | 6.83% |
| Production stream (2006–mid-2009) | 121,664 | 23,002 | 18.91% |

The rising positive rate across time-ordered splits is the financial crisis
arriving. The production stream is quarantined from training precisely because
it contains it.

---

## 2. Promotion: a better model, refused

The single most important piece of evidence in this project.

A drift check recommended retraining. The retrained candidate was trained on a
window ending 2009-06-25 (25,154 rows) and scored **validation average precision
0.734392 against production's 0.732031** — genuinely better. It was rejected.

```
gates    5 of 7 passed, 2 blocking failures

  [FAIL] min_improvement   average_precision improved by only 0.002361,
                           below the required 0.010000
  [FAIL] reproducibility   the working tree had uncommitted changes
  [PASS] minimum_metric    0.734392 >= 0.250000
  [PASS] roc_auc_regression 0.972972 >= 0.973004 - 0.005000
  [PASS] calibration       brier 0.022007 <= 0.022446, skill > 0
  [PASS] recall_regression 0.876707 >= production - 0.000000
  [PASS] latency           mean prediction <= 1.0000 ms

result   REJECTED — production unchanged, nothing registered
```

Source: `artifacts/benchmarks/promotion-candidate-20260912T075006Z.json`.

Two things are worth separating. The `min_improvement` failure is the system
working as designed: +0.0024 is inside the noise band the gate exists to filter,
and churning production for it costs more than it wins. The `reproducibility`
failure is an honest artifact of running the retrain mid-development with an
uncommitted tree — and it is also the gate doing its job, because a candidate
that cannot be tied to a committed revision cannot be reproduced later.

### The denominator

The live `/platform/promotions` endpoint reports **15 candidates trained, 1
registered** against this store. A registry shown on its own makes it look as
though everything ever trained was promoted; the count of runs that never
reached it is what makes the gates mean something.

---

## 3. Drift

Measured as Population Stability Index per feature, against the training
reference distribution, over a window of the quarantined production stream.

Latest recorded check (`drift-20260912T075056Z`):

| | |
|---|---|
| Window | 2008-01-01 .. 2009-06-25 |
| Scenario | `lending_shift` |
| Reference | train split |
| Trigger rule | PSI ≥ 0.1 on ≥ 3 features |
| Result | **21 of 24 features over threshold** |
| Max PSI | 4.152432 |
| Decision | `retrain` |

Top movers:

| Feature | PSI |
|---|---|
| `disbursement_ratio` | 4.1524 |
| `sba_guarantee_ratio` | 4.0139 |
| `term_bucket` | 3.9916 |
| `Term` | 3.4177 |
| `term_years` | 3.4177 |
| `NoEmp` | 3.1798 |
| `LowDoc` | 2.0976 |
| `naics_sector` | 0.8596 |

A PSI above 0.25 is conventionally "major shift". Values above 3 are the 2008
credit contraction showing up in loan structure: terms, guarantee ratios and
disbursement ratios all moved at once.

**What this does not show.** Drift is measured on inputs. It cannot say the
model became less accurate, and nothing in this repository claims it does — see
§7.

---

## 4. Canary and rollback

Current allocation is **0.0%**: no canary is running. Incumbent is v1.

The last recorded canary decision (`canary-20260912T091224Z`) was `hold`, with
the reason *"only 0 canary request(s); 50 needed to decide"*. That is the
evaluator refusing to conclude from insufficient traffic rather than reading
noise as signal, and it is the correct outcome for an idle canary.

Routing is a SHA-256 hash of the routing key into 10,000 buckets, so a given
caller lands on the same tier every time. Traffic is capped at 99%: a canary can
never be given all traffic, because then there is no incumbent left to compare
against or fall back to.

### Rollback signals

Error rate (absolute and relative to the incumbent), upstream failure rate,
latency p95 (absolute and relative), and candidate serving health.

**Accuracy is deliberately excluded.** Labels on this dataset mature after 60
months; a canary waiting for accuracy would never conclude. See §7.

### Observed under failure

From the `canary_failure` scenario, with 40% allocation and the canary scaled to
zero: **7 of 20 requests failed** (40% allocation, small-sample), and after
rollback **0 of 20** reached the canary. Production was v1 before and after.

---

## 5. Failure engineering

Six scenarios. Five break real components with `kubectl scale`; one —
`bad_candidate_promotion` — is a controlled double and says so in its own report.
Every injector has a matching restore in a `finally`.

Recorded live run:

| Scenario | Observed | Invariants |
|---|---|---|
| `model_serving_failure` | `/predict` 503 with no scores, `/ready` 503, `/health` 200; recovery returned 0.003955 | 3/3 |
| `mlflow_dependency_failure` | `/predict` 200 with 8 scores; a registry lookup returned `None`; model identity unchanged after recovery | 3/3 |
| `canary_failure` | 7 of 20 failed at 40% allocation; 0 of 20 after rollback; production v1 throughout | 3/3 |
| `telemetry_failure` | 12 of 12 requests served normally; `/metrics` 200; `/ready` 200 | 2/2 |
| `restart_recovery` | `/predict` 200 with 0.003955 (was 0.003955); canary allocation 0.0% → 0.0% | 4/4 |
| `bad_candidate_promotion` *(controlled double)* | 1/7 gates passed, 6 blocking failures; production v1 → v1, nothing registered | 2/2 |

Full detail, including the invariant definitions and the two defects the harness
found in itself, is in [failure.md](failure.md).

The governing idea: **failing closed is success.** A service answering 503
because its model tier is gone has behaved correctly. One answering 200 with a
number it invented has not — and a loan decision made on a fabricated score is
indistinguishable from a correct one until much later.

---

## 6. Reproducibility

`python -m ml_platform reproduce` retrains from recorded provenance and compares
every metric.

| | |
|---|---|
| Profile | `strict` (1e-6 on every metric) |
| Metrics compared | 48 |
| Exact matches | **48** |
| Deviations | none |
| Split mismatches | none |

Recorded environment: CPython 3.12.14, Windows AMD64, pandas 3.0.5, numpy 2.5.2,
scikit-learn 1.9.0, scipy 1.18.1, pandera 0.33.1, joblib 1.6.0, seed 42, one
thread, `quicksort` row ordering.

Provenance: git `4055124be9d9`, **`git_dirty: false`**, lockfile
`7e852228999ea8b3`, dataset `0359128a0b75`.

Source: `artifacts/benchmarks/reproducibility-baseline-20260912T191935Z.json`.

A cross-platform run has not been attempted. That is what the `portable` profile
(5e-4 on ranking metrics, 2e-3 on threshold-derived ones) exists for, and it
remains untested — see the CI note below.

---

## 7. Three clocks — what this platform can and cannot know

The distinction the rest of this document depends on.

| Signal | Latency | What it may do |
|---|---|---|
| **Input drift** | Immediate | May trigger retraining. Never rejects, never rolls back, never promotes. |
| **Operational health** | Seconds | May roll back a canary. Never promotes. |
| **Realised performance** | ~60 months | Would be the real answer. Is not available. |

The label is "defaulted within 60 months of disbursement". A loan disbursed
today cannot be labelled until 2031. So:

- No realised-performance signal exists in this system, and none is simulated.
- Drift triggering a retrain is a hypothesis, not a verdict. The retrained
  candidate still faces the same seven gates (§2), and in the one recorded case
  it lost.
- Rollback uses operational signals only. This is not a simplification — it is
  the only decision available on a timescale where rolling back is still useful.

Nothing on the dashboard, in the API, or in these docs claims to know whether
the production model has become less accurate.

---

## 8. Test suite and static checks

| | |
|---|---|
| Backend tests | 899 passing |
| Frontend tests | 33 passing |
| Ruff (`check` and `format`) | clean on `src` and `tests` |
| mypy | clean, 58 source files |
| Source files | 58 Python modules, 41 test modules, 15 dashboard TS/TSX |

### Continuous integration

Four jobs are defined in `.github/workflows/ci.yml`: lint/format/types, the test
suite, a reproducibility run against the real register on the `portable` profile,
and a Docker build with a smoke test.

**CI has never run.** This repository has no remote, so the workflow is defined
and unexercised. Everything reported above was run locally on Windows. That also
means the `portable` tolerance profile — the one that exists for a rerun on
different hardware — has never been exercised, because CI is the only place that
would use it.

Every failure invariant is tested in both directions — an invariant that cannot
fail proves nothing. The gate and invariant catalogues served to the dashboard
are asserted equal to `promotion.gates.GATE_BUILDERS` and
`failure.invariants`, so the dashboard cannot describe a system that no longer
exists.

---

## 9. Limitations

Stated plainly, because a portfolio system that hides these is less honest than
one that has fewer features.

- **No realised performance, ever, in this deployment.** See §7. The lifecycle is
  complete in machinery and permanently incomplete in evidence.
- **Two registries, not synchronised.** The laptop's SQLite store and the
  cluster's tracking server hold different histories. Moving a model between
  them is manual and deliberate; nothing reconciles them.
- **Six failure scenarios, not a fault model.** Nothing tests partial network
  partitions, disk exhaustion, slow-but-alive dependencies or corrupted
  artifacts. Scaling to zero is a clean failure; the messy ones are harder to
  inject and were not attempted.
- **Nothing runs the failure scenarios on a schedule.** They are a command.
- **KServe runs in RawDeployment mode**, which has no traffic-splitting
  primitive, so the canary split is done in the application tier. This is a real
  architectural constraint, not a design preference — see [kserve.md](kserve.md).
- **A pod that starts while MLflow is down stays unready** until restarted. It
  fails closed, so this is an availability limitation rather than a safety one,
  and it is recorded rather than papered over — see [failure.md](failure.md).
- **Single-node cluster on a laptop.** No multi-zone behaviour, no real load, no
  noisy neighbours. Latency numbers are indicative only.
- **The `portable` reproducibility profile is untested, and CI has never run.**
  The workflow is defined; the repository has no remote. Only same-platform
  `strict` reproduction has been verified, on Windows.
- **The canary evaluator has never seen real canary traffic** — the one recorded
  decision was `hold` for insufficient requests. Its logic is tested; its
  judgement under load is not.
