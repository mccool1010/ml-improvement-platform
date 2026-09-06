# Data

## Source

The U.S. Small Business Administration national loan guarantee register, covering
7(a) and 504 program approvals. It is the dataset published alongside Li, Mickel &
Taylor (2018), *Should This Loan be Approved or Denied?*, Journal of Statistics
Education 26(1).

| Property | Value |
|---|---|
| Rows | 899,164 |
| Columns | 27 |
| Approval dates | 1961-12-07 to 2014-06-25 |
| File size | 179,430,516 bytes |
| SHA-256 | `0359128a0b7599e83e4c2e4dcdd781d9121a765237f98d9f0ec1ab3e7c522548` |

The file is not committed. `scripts/download_data.py` fetches it and verifies the
checksum, and every training run records that checksum. A changed upstream mirror
fails the run rather than silently altering results.

## The observation cutoff

`2014-06-25`, the last approval date in the register, acts as "today" for the whole
project. Nothing after it may be treated as observed. This is what makes the
delayed-label design testable rather than aspirational.

## Target

`target` is 1 if the loan charged off within **60 months of disbursement**, else 0.

A row is included only once those 60 months have fully elapsed before the
observation cutoff. That rule depends only on the calendar, never on the outcome,
so both classes are filtered identically and the label is unbiased.

After preparation, **682,428 rows** are usable, with an overall positive rate of
**5.80%**. That is 847,077 rows with an elapsed horizon, then restricted to
uniform exposure as described below.

## Uniform exposure

A second restriction keeps only loans whose scheduled term covers the whole
horizon, meaning `Term >= 60` months.

Without it the target is contaminated by how long each loan was even capable of
defaulting. A 36-month loan lives its entire life inside a 60-month window; a
240-month loan is a quarter of the way through. The effect is not subtle. On the
unrestricted data the default rate cliffs at exactly the horizon boundary:

| Term bucket | Rows (test 2005) | Default rate |
|---|---|---|
| 1-3 years | 9,079 | 40.8% |
| 3-5 years | 14,394 | 41.0% |
| **5-7 years** | 31,170 | **8.3%** |
| 7-10 years | 7,322 | 9.0% |
| 10-20 years | 6,347 | 2.9% |

Measured directly, `Term` alone then reproduced 96% of full-model performance
(0.7211 average precision against 0.7535 for every feature together). The model
was largely learning the shape of the observation window rather than credit risk.

Restricting to `Term >= 60` costs 19.4% of rows and removes the artifact. `Term`
remains a genuine and strong signal afterwards, because long-dated loans in this
register are collateralised real-estate lending and default far less than
short working-capital loans.

### Why not "did it eventually charge off"

Because the two classes resolve on different clocks, and filtering on resolution
therefore biases the sample.

A charge-off resolves when it happens. Measured on this register, the median gap
between disbursement and charge-off is 1,410 days, with the tenth percentile at
655 days and the ninetieth at 2,651 days. A healthy loan resolves only at
scheduled maturity, which for a 240-month loan is twenty years out.

Filtering to loans resolved by the cutoff therefore keeps far more of the defaults
than of the healthy loans, and worsens for recent cohorts:

| Approval FY | Share retained by a resolution filter | Reported default rate |
|---|---|---|
| 2006 | 84.8% | 41.0% |
| 2007 | 74.7% | 56.7% |
| 2008 | 58.7% | 69.1% |
| 2010 | 26.4% | 49.4% |

A 69% default rate for 2008 is an artifact of the filter, not a fact about
lending. The fixed-horizon label removes it.

### What the horizon costs

Sixty months captures 68.9% of eventual charge-offs. The remainder occur later and
are labelled negative. This is a stated limitation, not a hidden one, and every
metric in the project is conditioned on it. The horizon is a configuration value.

| Horizon | Share of eventual charge-offs captured |
|---|---|
| 24 months | 13.3% |
| 36 months | 32.4% |
| 48 months | 52.7% |
| **60 months** | **68.9%** |
| 84 months | 88.4% |

Sixty months was chosen because it captures a clear majority of defaults while
still leaving approvals through mid-2009 usable, which keeps the entire financial
crisis inside the analysable window.

## Splits

Strictly ordered by approval date, contiguous, non-overlapping, never shuffled.

| Split | Approval window | Rows | Positives | Rate |
|---|---|---|---|---|
| train | 2000-01-01 to 2003-12-31 | 150,160 | 4,602 | 3.06% |
| validation | 2004-01-01 to 2004-12-31 | 54,285 | 2,587 | 4.77% |
| test | 2005-01-01 to 2005-12-31 | 53,487 | 3,712 | 6.94% |
| production_stream | 2006-01-01 to 2009-06-30 | 121,664 | 23,266 | 19.12% |

`production_stream` is not used during model development. It is replayed as live
traffic for drift detection, delayed-label evaluation and retraining.

Training starts in 2000 because `UrbanRural` is effectively unpopulated before
then. Its mean value is 0.01 in 1998 and 0.44 in 1999, against 1.14 from 2000
onward, meaning the field was being filled in during those years. Including that
period would train the model on a schema that no longer exists.

## Drift

The drift in this dataset is real and has a known cause. Default rate within 60
months, by approval year, on the uniform-exposure population:

| Year | Rows | Rate | | Year | Rows | Rate |
|---|---|---|---|---|---|---|
| 2000 | 29,895 | 2.50% | | 2005 | 53,487 | 6.94% |
| 2001 | 31,481 | 2.87% | | 2006 | 51,978 | 13.96% |
| 2002 | 39,036 | 2.95% | | 2007 | 44,532 | 23.89% |
| 2003 | 49,748 | 3.62% | | 2008 | 21,318 | 22.09% |
| 2004 | 54,285 | 4.77% | | 2009 | 3,836 | 17.31% |

The rate rises roughly tenfold from 2000 to 2007 and then begins to recover.

Feature distributions move too. Mean loan term falls from 132 months in 1998 to 78
in 2007 before recovering, and median employee count falls from 5 to 3 over the
same period. No synthetic corruption is used anywhere in this project.

## Known data quality issues

All of these are handled explicitly in `src/ml_platform/data/preprocessing.py` and
asserted in `src/ml_platform/data/validation.py`.

| Issue | Handling |
|---|---|
| Two-digit years spanning 1961 to 2014 | `strptime` pivots `%y` at 1969, so `07-Jan-62` parses as 2062. Any year after a fixed pivot of 2059 is pulled back a century. |
| Currency as `"$60,000.00 "` | Stripped of symbols and whitespace, then coerced to float. |
| `ApprovalFY` contains `1976A` | Four-digit year extracted by regex. |
| `RevLineCr` and `LowDoc` hold stray codes beyond Y/N | Mapped to `{Y, N, UNK}`. `UNK` is a real level, not a dropped row. |
| `NewExist` coded 0 or missing | Restricted to `{1, 2}`, otherwise null. |
| `NAICS` is 0 for 22.5% of rows | Mapped to sector `unknown`, kept as its own level. |
| `DisbursementDate` missing for 0.26% | Filled with `ApprovalDate`. |
| `Term` is 0 for 810 rows | Visible to the schema check, then removed by the uniform-exposure restriction. |
| 1,997 rows have no `MIS_Status` | Excluded; they have no label at all. |

## A labelling bug found at M3

The century correction originally pivoted on the observation cutoff of
2014-06-25, the last approval date in the register. That was wrong. Charge-offs
follow the approvals they relate to, so a charge-off recorded in August 2014
legitimately post-dates the last approval.

The consequence was not a cosmetic date error. 1,491 real mid-2014 charge-off
dates were rewritten to 1914, which made the time from disbursement to
charge-off strongly negative. A negative interval satisfies "charged off within
60 months", so **1,096 loans were labelled as in-horizon defaults when their
charge-off actually fell outside the horizon.**

The fix pivots on a fixed year, 2059, chosen because `strptime` maps two-digit
years 60 to 68 into 2060 to 2068 while nothing in the file is legitimately later
than the mid-2010s. The prepared schema now also asserts that every date column
falls between 1960 and 2020, so this class of fault fails validation rather than
producing a plausible-looking model.

The effect on results was small and changed no conclusion. Usable rows fell from
682,428 to 682,421, the test-split default rate from 6.94% to 6.83%, baseline
test average precision from 0.1632 to 0.1629 and candidate from 0.7012 to 0.7036.
The reference in `configs/reference.yaml` was re-locked, with the reason recorded
in the file.

## Leakage

These columns are known only after the lending decision and are dropped before
modelling:

- `MIS_Status`, the raw outcome the target derives from
- `ChgOffDate`, the date of the outcome
- `ChgOffPrinGr`, non-zero for 99.6% of charge-offs and 0.7% of healthy loans
- `BalanceGross`, an outstanding balance recorded at resolution

Free-text identifiers (`LoanNr_ChkDgt`, `Name`, `City`, `Zip`, `Bank`) are also
dropped. They carry entity identity rather than credit risk.
