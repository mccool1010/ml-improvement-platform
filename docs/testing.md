# Testing

The suite exists to protect the assumptions that make the results meaningful, not
to accumulate a test count. Every test here fails for a reason someone would care
about: a biased label, a leaked outcome, a metric that silently moved, or a
repository that cannot be built from a clean clone.

## Tiers

| Tier | Location | What it protects |
|---|---|---|
| Unit | `tests/unit/` | Each stage in isolation: configuration, determinism, ingestion, validation, preprocessing, splitting, features, models, evaluation |
| Data validation | `tests/unit/test_validation.py` | Schema, types, required fields, ranges, invalid category codes, the label horizon invariant |
| Regression | `tests/regression/` | The three M0 findings, the locked M1 reference, and repository completeness |
| Integration | `tests/integration/` | The whole training path wired together, artifacts, and the CLI contract |

## The assumptions under protection

**The label is a fixed 60-month horizon, not eventual outcome.**
`tests/regression/test_label_construction.py`. The rejected definition is biased,
because defaults resolve in a median 1,410 days while healthy loans resolve at
maturity. The key test asserts that observability depends only on the calendar,
never on the outcome, so both classes are filtered identically.

**The population has uniform exposure.**
`tests/regression/test_term_leakage.py`. This is the subtlest finding and the
easiest to reintroduce, because reintroducing it improves every headline metric.
The tests build a cohort where every loan shares one hazard rate, show that
`Term` still predicts the label purely through the observation window, and then
assert the invariant the restriction buys: after it, every loan is observable for
exactly the same 60 months.

Note what those tests deliberately do not claim. The restriction does not flatten
the relationship between `Term` and the label, and it should not. Within the
restricted population term genuinely proxies product type, since long-dated SBA
lending is collateralised real estate. What the restriction removes is the
discontinuity at the horizon boundary.

**No feature reaches forward in time.**
`tests/unit/test_features.py` asserts that neither feature set declares a
post-decision column, and that outcome columns present in the source frame still
cannot reach the model matrix.

**The metrics have not moved.**
`tests/regression/test_reproducibility.py`. The comparison machinery is tested on
synthetic records so its failure paths can be exercised in milliseconds; the full
reproduction against the genuine dataset is marked `slow`.

**The repository can be built from a clean clone.**
`tests/regression/test_packaging.py` inspects what git actually tracks. It exists
because an unanchored `.gitignore` pattern once excluded the entire modelling
package while every local test still passed.

## Running

```bash
pytest                    # everything
pytest -m "not slow"      # skips the tests that read the full dataset
pytest tests/unit         # one tier
```

Tests marked `slow` need the real 682,428-row register and skip automatically when
it has not been downloaded, so a fresh clone gets a green run without a 179 MB
download.

## The synthetic register

Integration tests run against a generated dataset shaped exactly like the real
file, built in `tests/conftest.py`. It reproduces the awkward parts deliberately:
two-digit dates, currency as `"$60,000.00 "`, stray codes in the Y/N columns, a
non-numeric fiscal year, and a rising default rate over time. Only the data is
substituted; every stage of code under test is the production one.

Two properties of the fixture are load-bearing. Charge-off dates never post-date
the observation cutoff, because the register only records events that had already
happened, and a later date would be pulled back a century by the correction and
produce 1914 timestamps. And the risk drivers survive the exposure restriction, so
a model trained on it has real signal to learn; a fixture whose only signal lived
in short-term loans would leave the integration tests unable to tell a working
pipeline from a broken one.
