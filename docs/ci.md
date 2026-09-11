# Continuous integration

Every claim this project makes is checkable by running something. CI runs those
same commands on a machine that is not the author's, on every push to `main` and
every pull request against it. Nothing here deploys anything, and nothing here
holds a credential.

Definition: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## The four jobs

| Job | What it proves | Command |
|---|---|---|
| `static` | The code lints, is formatted, and type-checks under `mypy --strict` | `ruff check .`, `ruff format --check .`, `mypy` |
| `tests` | The whole suite passes on a clean checkout | `pytest` |
| `reproducibility` | The locked M1 reference metrics still reproduce on the real 682,421-row register | `python -m ml_platform reproduce --profile portable` |
| `container` | The M8 image builds, starts, and reports its own state honestly | `docker build`, then `/health` and `/ready` |

They are separate jobs so a red tick names its own cause. A lint failure and a
labelling regression are not the same event and should not look the same.

## Reproducing CI locally

The commands are the same ones, in the same order. On Windows:

```
uv sync --frozen --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python -m ml_platform download
uv run python -m ml_platform reproduce --profile strict
docker build -f docker/Dockerfile -t ml-platform-api:local .
```

Where `make` is available, `make check` wraps the four static-and-test commands
and `make verify` adds the reproduction.

The one deliberate difference is the tolerance profile. Locally, on the platform
the reference was locked on, use `strict`, which the project has verified to be
bit-exact. CI uses `portable`.

## Why CI uses the `portable` profile

`configs/reference.yaml` defines two tolerance profiles, and has since M2:
`strict` for a rerun on the reference platform, `portable` for "a rerun on
different hardware or a different operating system". The runners are Linux; the
reference was locked on Windows. A different BLAS build can reorder a
floating-point reduction and move the last bits of a sum, and that is exactly the
case `portable` was written for. Using it in CI is the contract working as
designed, not a relaxed gate.

What `portable` does **not** loosen is the part that catches real faults. Split
row counts, positive counts, the dataset checksum and the row-order fingerprint
are compared **exactly** under either profile. A labelling change, a sort-order
change, or a swapped dataset fails CI just as hard as it fails locally. The
tolerance applies only to the metric values, and only at 5e-4.

If CI ever fails the `portable` bound, the correct response is to record a
platform-specific reference with its own evidence — not to widen the numbers
until the run goes green.

## What CI does not test, and why

**A containerised model load.** The `container` job builds the image, starts it,
and asserts that `/health` answers `ok` while `/ready` answers `model_loaded:
false`. It cannot assert a real prediction.

The reason is a genuine limitation found in M8, not an omission. This project
tracks with a local MLflow SQLite backend and a filesystem artifact root, and
MLflow records each experiment's artifact location in the database as an absolute
host URI — here, `file:///C:/mlops/mlartifacts`. Those rows are data, not
configuration. A Linux runner has no such path, so a container given that store
resolves an artifact to nothing and reports `No such artifact: ''`. Locally the
container is made to work by bind-mounting the artifacts at the literal recorded
path, which is a workaround for one machine and cannot be a CI step.

Rather than fake a model in CI, the job checks the property that is genuinely
true and genuinely worth protecting: **an image with no registry mounted must
start, stay alive, and refuse to declare itself ready.** That is the same
distinction `/health` and `/ready` exist to draw, and a regression that made an
unready service report ready would be caught here.

Serving a real promoted model in CI needs a tracking server with a shared
artifact store, reachable from a runner. That is infrastructure, and it belongs
to a later milestone. It is not worked around here.

Writing that job immediately earned its place. Run with no store mounted, the M8
image did not start and report itself unready at all: `/app` was root-owned, so
SQLite could not create the tracking database, and MLflow retried that failure
with exponential backoff *inside the startup lifespan*. The container sat there
with the port open and answered nothing, for minutes. Every earlier manual test
had mounted a store, so nobody had ever run it the way a runner would.
`docker/Dockerfile` now gives the service user ownership of its working
directory, and `tests/unit/test_container.py` keeps it that way.

**The `slow` tests, in the `tests` job.** They need the 179 MB register and skip
without it. The `reproducibility` job trains both models on that exact data, so
the coverage is not lost; downloading it in both jobs would only double the cost.

## Determinism and caching

The runner image is pinned to `ubuntu-24.04` and `uv` to an exact version. A
runner image that changed underneath the project would be indistinguishable from
a regression.

Native thread pools are pinned to one thread in the workflow environment, because
`pytest` imports numpy without passing through the CLI's `_bootstrap`. This is
the same contract `configs/base.yaml` sets and the Dockerfile repeats.

Three caches, all of them keyed on something immutable:

- the `uv` cache, keyed on `uv.lock`;
- the raw register, keyed on the SHA-256 recorded in `configs/base.yaml`, so the
  cache can only ever hold the bytes the reference was produced from — and the
  download step re-verifies that checksum whether or not the cache hit;
- Docker layers, via the GitHub Actions build cache.

## Secrets

There are none. `permissions` is `contents: read`, the image is built with
`push: false`, and the workflow references no `secrets.*` value. Adding a
registry push later means adding a credential, and that is a decision to take
deliberately rather than inherit.

## Protecting the workflow itself

`tests/unit/test_ci_workflow.py` reads the YAML and asserts what CI claims to do:
that it installs from the lockfile, runs each of the five checks, names a
tolerance profile the reference actually defines, keys the dataset cache on the
recorded checksum, requests no secret, and marks nothing `continue-on-error`.

A workflow that has quietly stopped running `mypy` still shows a green tick.
Nothing else in the project would notice.
