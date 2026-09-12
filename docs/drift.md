# Drift and retraining

Detecting that the incoming population has moved, retraining on it, and putting
the result to the existing quality gates.

```
drift check ──► retrain on the new window ──► M6 gates ──► registry
   (inputs)          (candidate)              (decides)    (only if passed)
```

Nothing in this path can change what is served. Retraining produces a
*candidate*; `run_promotion` decides, using the same seven gates and the same
validation split as `python -m ml_platform promote`.

## The three clocks, kept apart

M0 separated three signals, and this milestone is where that separation earns its
keep:

| Signal | Available | Authority |
|---|---|---|
| Input drift | immediately | may trigger a retraining *attempt* |
| Operational health | immediately | may trigger rollback (M14) |
| Realised performance | ~5 years on this dataset | the only thing that can say a model got worse |

Drift detection here reads **inputs only**. It produces no label, estimates no
accuracy, and cannot reject a model. A drift decision is `retrain` or
`no_action` — never `rollback`, and never "the model is bad". Conflating the
first clock with the third is the standard way a platform ends up retraining
itself into a worse model on the strength of a distribution shift that never hurt
anything.

## Methodology

Both frames go through the same `build_features` code the model trains on, so
drift is measured on what the model actually sees rather than on raw columns.
Two statistics per feature:

- **PSI** (Population Stability Index) over ten quantile bins for numeric
  features, or category proportions for categorical ones. **This is what drives
  the decision**, because it is an effect size.
- **KS** (numeric) or **chi-square** (categorical), reported alongside as
  corroboration. These are *not* the trigger: on 25,000 rows a significance test
  rejects almost any difference, so using one as a threshold would mean
  retraining forever.

Bin edges come from the **reference**, never from the combined data — otherwise
the current window moves the ruler it is being measured against. Categories
rarer than 1% in the reference are pooled, so a long tail cannot manufacture
drift out of noise, and unseen categories land in one `__other__` bucket.

### Thresholds

From `monitoring.drift` in `configs/base.yaml`:

| Setting | Default | Why |
|---|---|---|
| `threshold_psi` | `0.1` | The conventional "noticeable shift" reading. 0.25 is "different population" and is reported as `major`. |
| `min_drifted_features` | `3` | One feature can move for a mundane reason — a changed code list, a seasonal effect. Requiring several is what makes this a population-level statement instead of a hair trigger. |
| `reference_split` | `train` | The distribution the production model is held to have learned. |
| `window` | `2008-01-01 .. 2009-06-25` | A slice of the quarantined production stream. |

### Is the detector calibrated?

Two checks that matter more than any single number. Against the real register:

```
reference vs itself                  0/24 drifted   max PSI 0.0000  -> no_action
train vs a 2002-03 slice of itself   1/24 drifted   max PSI 0.6323  -> no_action
train vs validation (2004)           7/24
train vs test (2005)                12/24
train vs 2006                       17/24
train vs 2008-2009                  19/24   max PSI 2.0976  -> retrain
```

Drift rises **monotonically with distance in time**, and the identity case is
exactly zero. A detector that cannot say "no" is worthless, and this one can.

The single feature flagged in the 2002-03 self-comparison is `naics_sector`, and
it is real: NAICS coding was phased into this register across 1998-2003, so the
early years carry far more missing codes.

## The current window, and the controlled scenario

The honest default is a real date slice of the **2006 to mid-2009 production
stream** that M0 quarantined and no model has trained on. Slicing is
deterministic, so the same dates always give the same rows, and each window
carries a content-and-order `fingerprint` that ties a drift decision to exact
rows.

That real window already drifts hard — it spans the financial crisis. A
**controlled scenario** exists alongside it because a demonstration needs a
signal that can be dialled to a known size, and a test that depends on the crisis
being severe enough is a test that fails for the wrong reason. Scenarios live in
`ml_platform.monitoring.windows` and are named, seeded transformations of a real
window:

| Scenario | What moves |
|---|---|
| `none` | nothing — the real window, the control |
| `mild` | loan sizes +6% — below the threshold on purpose, to show the detector does not fire on any change at all |
| `lending_shift` | amounts +70%, guarantee ratios toward the cap, longer terms, fewer employees, more new/urban businesses — the shape of an underwriting policy change |
| `new_categories` | unseen NAICS codes and a concentrated state mix |

Two rules the scenarios obey, both enforced in code rather than by convention:

**Only inputs are perturbed.** `target`, `MIS_Status`, `ChgOffDate`,
`ChgOffPrinGr`, `BalanceGross` and the dates are passed through untouched, and
`apply_scenario` raises if a scenario modifies one. Manufacturing an outcome to
match manufactured inputs is fabricating ground truth, which is the one thing
this milestone must not do.

**The perturbation is a pure function of seed and window.** No clock, no global
random state, and the scenario name contributes through a stable digest rather
than `hash()` — which is randomised per process and would have made the "same"
window differ between runs.

## Which labels retraining uses — read this one

Retraining trains on production-window rows **that have labels**. They have them
because this is a historical register whose 60-month horizon has fully elapsed
for those approvals: matured labels, not invented ones.

In a live system those labels would not exist yet. A loan approved today cannot
be known to have charged off within five years until five years have passed. So
what this milestone demonstrates is the *mechanism* — drift triggers retraining,
retraining is judged by gates — on a dataset where the waiting has already
happened.

`assert_labels_matured` enforces the line rather than trusting it, and it is
checked against the rows actually being trained on, not the dates that were
requested. `monitoring.label_maturity_end` is `2009-06-25`: the observation
cutoff minus the label horizon. **Writing this check is what caught the first
version of the configured window ending 2009-06-30**, five days past the point
where an outcome could be known.

## Retraining, and what it deliberately does not change

`run_training` gained one inert parameter, `extra_training_frame`. It appends
rows to the **training split and nothing else**. Absent, every existing run is
unchanged and still bit-exact.

The evaluation splits are deliberately untouched. `build_comparison` reads the
incumbent's metrics from its **registry tags** — the numbers recorded when *it*
was promoted — so moving the validation window would compare a candidate on new
rows against an incumbent on old rows, and silently weaken the gate. Both models
are judged on the same 2004 validation rows.

The cost, stated plainly: a retrained model trains on data from 2006-2009 and is
evaluated on 2004. That is not leakage — different loans, and
`_assert_disjoint_from_evaluation` fails loudly if the window ever overlaps
validation or test — but it is a weaker proxy for how the model will do on the
period it will actually serve. The principled fix is to roll the evaluation
windows forward *and* re-evaluate the incumbent on them, which changes M6
machinery. That is deferred, deliberately, and recorded here.

## Traceability

Every artefact carries a `drift_event_id`, so a promotion months later can be
traced back to the check that prompted it:

```
drift run (run_type=drift_check)          tags: drift_event_id, decision, window fingerprint
  -> candidate (run_type=retraining_candidate)  tags: drift_event_id, retrain_window_*
     -> registered version                       tag: mlflow_run_id -> the candidate run
```

The registry link needs no change to M6: a registered version already records its
`mlflow_run_id`, and that run carries the drift tags.

## Runbook

```bash
python -m ml_platform drift                          # check, write a report
python -m ml_platform drift --scenario lending_shift # controlled demonstration
python -m ml_platform retrain                        # check, and retrain if drifted
python -m ml_platform retrain --force --no-register  # scheduled refresh, gates only
```

`drift` exits 0 for `no_action` and 2 for `retrain`, so a scheduler can branch on
it. Two is not a failure — drift is a finding.

Reports land in `artifacts/reports/drift-*.json` (per-feature) and
`artifacts/benchmarks/retrain-*.json` (the decision).

## What happened when this was run for real

```
drift    19/24 features over psi=0.1, max 2.0976 (LowDoc)     -> retrain
retrain  candidate validation AP 0.734392 vs production 0.732031
gates    [FAIL] min_improvement   +0.002361 < required +0.010000
         [FAIL] reproducibility   the working tree had uncommitted changes
         [PASS] minimum_metric, roc_auc_regression, calibration,
                recall_regression, latency
result   REJECTED - production unchanged, nothing registered
```

The retrained model was *better*, and was still refused: the improvement was
inside the noise band the gate exists to filter out. That is the system working.

## Limitations

- **The gates decide on 2004 validation rows** while retraining uses 2006-2009
  data, for the reason above. Rolling evaluation forward needs the incumbent
  re-scored, which is an M6 change.
- **No scheduling.** `drift` and `retrain` are commands. Running them on a timer
  is a CI/deployment concern and is not wired up here.
- **No performance-degradation trigger.** Only input drift can trigger
  retraining, because realised performance takes five years to arrive on this
  dataset. That is the correct behaviour, not a missing feature.
- **The drift report is not on the Grafana dashboard.** The event is in MLflow
  with per-feature PSI metrics; surfacing it next to the serving panels would
  mean exporting it to Prometheus, and a gauge that updates when someone runs a
  command is a poor fit for a scrape-based system.
