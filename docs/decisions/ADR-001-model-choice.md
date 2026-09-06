# ADR-001: Problem, dataset, label, and model family

- **Status:** Accepted
- **Date:** 2026-09-06
- **Milestone:** M0

## Context

The platform needs a supervised tabular problem that can carry the full improvement
loop. The binding requirement is not modelling difficulty. It is that the dataset
must support an honest demonstration of delayed ground truth and of real
distribution drift, because those two things drive retraining, and retraining is
the point of the project.

Candidate datasets were assessed against six criteria: a genuine calendar date, a
meaningful class imbalance, enough rows for stable measurement, headroom for a
baseline to be improved, real drift in a later period, and a licence and mirror
that allow unauthenticated reproduction.

## Decision

Use the **U.S. Small Business Administration national 7(a)/504 loan guarantee
register**, predicting whether an approved loan will charge off within 60 months
of disbursement.

Baseline: **logistic regression** on core register fields.
Candidate family: **histogram gradient boosting** on engineered features.

## Alternatives considered

**Lending Club accepted loans (2007-2018), rejected.** It has calendar dates and
more rows, but its delayed-label problem and its drift period are the same period.
Loans issued near the end of the file are mostly still current, so the only way to
obtain labels for the recent, drifted cohort is to wait, which the data does not
allow. Restricting to resolved loans introduces exactly the bias described below.
The file is also 1.68 GB against 179 MB, which makes continuous integration
impractical, and roughly a dozen of its columns are post-origination outcomes.

**Home Credit Default Risk, Give Me Some Credit, Taiwan credit card default,
rejected.** None carries an absolute calendar date. Home Credit expresses time
only as day offsets relative to each application, so no time-ordered split across
applications is possible.

**IEEE-CIS fraud detection, rejected.** Time-ordered splitting is possible and the
imbalance is strong, but the features are anonymised. Feature engineering cannot
be motivated by domain reasoning, which removes one of the two levers the project
needs to demonstrate model improvement.

**ULB credit card fraud, rejected.** It spans two days, so it cannot show drift.

## Why the SBA register fits

The register holds 899,164 approvals from 1961-12-07 to 2014-06-25, with 27
columns. After cleaning and labelling, 847,072 rows have a fully elapsed 60-month
horizon.

- **Real drift with a known cause.** The charge-off rate within 60 months moves
  from 4.4% for fiscal 2000 approvals to 36.7% for fiscal 2007, then back to 22.8%
  by fiscal 2009. This is the financial crisis passing through a lending book. No
  data corruption is needed to demonstrate drift.
- **Delayed ground truth is intrinsic, not simulated.** The register records
  `ChgOffDate` for 157,511 of 157,558 charge-offs. The median gap between
  disbursement and charge-off is 1,410 days, and the tenth percentile is 655 days.
  A label genuinely does not exist for years after the decision.
- **Interpretable features.** Term, employee count, industry code, guarantee
  amounts, lender state and urban/rural status support domain-motivated feature
  engineering.
- **Genuine messiness.** Currency fields arrive as `"$60,000.00 "`. Dates use
  two-digit years across a span that crosses 1969. `RevLineCr` and `LowDoc` are
  nominally Y/N but contain stray codes. `ApprovalFY` contains `1976A`. `UrbanRural`
  is effectively unpopulated before 2000. This gives the validation layer real work.

## The labelling decision, and the trap it avoids

The obvious label is "did this loan eventually charge off". It is wrong here.

Filtering to loans that have *resolved* by the end of the register biases the
sample, because the two classes resolve on different clocks. A charge-off resolves
when it happens, typically inside four years. A healthy loan only resolves at
scheduled maturity, often ten or twenty years out. Filtering therefore keeps a
much larger share of the defaults than of the healthy loans, and the distortion
grows for recent cohorts. Measured directly, that filter retains 85% of fiscal
2006 approvals but only 26% of fiscal 2010, and reports a fiscal 2008 default rate
of 69%, which is not credible.

Instead the project uses a **fixed 60-month horizon**:

- A loan is positive if it charged off within 60 months of disbursement.
- A loan is included only once 60 months have elapsed before the observation
  cutoff of 2014-06-25.

The inclusion rule depends only on the calendar, never on the outcome, so the
label is unbiased. The cost is that 68.9% rather than 100% of eventual charge-offs
are captured, and usable approvals end in mid-2009. Both are acceptable, and the
horizon is a configuration value rather than a constant in code.

## The exposure decision

A fixed horizon introduces a second problem that the labelling rule alone does not
solve. "Charged off within 60 months" is partly a question of whether the loan was
outstanding that long. A 36-month loan lives its whole life inside the window; a
240-month loan is a quarter of the way through.

Measured on the holdout, the default rate cliffs at exactly the horizon boundary:
41.0% for three-to-five year loans against 8.3% for five-to-seven year loans. With
the full population, `Term` alone reproduced 96% of full-model performance, 0.7211
average precision against 0.7535 for all features together. The model was largely
learning the shape of the observation window.

The population is therefore restricted to **`Term >= 60` months**, so every loan is
at risk for the same 60 months and none matures early. This costs 19.4% of rows,
leaving 682,428. `Term` remains a strong signal afterwards for a legitimate reason:
long-dated loans in this register are collateralised real-estate lending and
default far less than short working-capital loans.

The decision defines the modelled population as SBA term lending of five years or
more, which is a coherent business segment rather than an arbitrary cut.

## Consequences

- Every metric in the project is conditioned on a 60-month horizon and on the
  five-year-plus population, and reports must say so. A charge-off in month 61
  counts as a negative, and short-term working-capital loans are out of scope.
- Model development is confined to approvals up to 2005. Everything from 2006
  onward is reserved as a production stream.
- The base rate moves substantially between splits, so absolute average precision
  is not comparable across them. Lift over the base rate is reported alongside.
- The 60-month horizon means a real deployment of this model would wait five years
  for a complete performance readout. That is the honest constraint the monitoring
  design in ADR-003 has to work within.
