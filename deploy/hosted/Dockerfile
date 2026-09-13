# Hosted demo image: the platform image plus a self-contained model registry.
#
# The cluster deployment talks to an MLflow tracking server. A single hosted
# machine has none, so this image builds an ordinary SQLite registry at build
# time from deploy/hosted/bundle -- the production model and the run history the
# local store recorded (see export_bundle.py) -- and the API loads the model
# in-process. The stages below are docker/Dockerfile's, repeated so this file
# builds from a clean checkout on any host with no pre-built base image.
#
#   docker build -f deploy/hosted/Dockerfile -t sba-mlops-demo .
#   docker run -p 8000:8000 sba-mlops-demo      # http://localhost:8000

# ---- dashboard -------------------------------------------------------------
# Built here rather than on the developer's machine so the image is
# self-contained: `docker build` alone produces a working /dashboard, with no
# "remember to run npm first" precondition that would silently ship a stale
# bundle or none at all.
FROM node:22-slim AS dashboard

WORKDIR /dashboard

# npm ci installs exactly the committed lockfile, the same contract uv.lock
# carries on the Python side.
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci

COPY dashboard/tsconfig.json dashboard/vite.config.ts dashboard/index.html ./
COPY dashboard/src/ ./src/
RUN npm run build

# ---- builder ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is rebuilt only
# when the lockfile changes, not on every source edit.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# The project itself, installed into the same environment.
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# curl is present for the container healthcheck below and nothing else.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# The static bundle only. The API mounts it at /dashboard when it is present
# and skips the mount when it is not, so this COPY is the only thing that
# decides whether the image serves a dashboard.
COPY --from=dashboard --chown=appuser:appuser /dashboard/dist/ ./dashboard/dist/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    # Native thread pools are pinned here as well as in the CLI, because uvicorn
    # imports the app directly and never passes through `_bootstrap`.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

# Writable mount points for the MLflow store. Owned by appuser so a bind mount
# or volume works without running as root.
#
# /app itself must be writable too. The tracking backend is a SQLite file at
# /app/mlflow.db, and SQLite creates its database and journal files in the
# directory. Left root-owned the store cannot be opened, and MLflow retries
# that failure with exponential backoff *inside the startup lifespan*, so a
# container with no store mounted never serves /health at all instead of
# starting and reporting itself unready. Found by running the image with no
# mounts, which is exactly what CI does.
RUN mkdir -p /app/mlartifacts /app/artifacts/reports /app/artifacts/benchmarks \
    && chown -R appuser:appuser /app/mlartifacts /app/artifacts \
    && chown appuser:appuser /app

USER appuser

EXPOSE 8000

# Readiness is the honest signal here: the process can be healthy while holding
# no promoted model. The healthcheck therefore asks /health, and an orchestrator
# gates traffic on /ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# ---- hosted registry --------------------------------------------------------
COPY --chown=appuser:appuser deploy/hosted/bundle/ /app/bundle/
COPY --chown=appuser:appuser deploy/hosted/build_store.py /app/build_store.py

RUN python /app/build_store.py /app/bundle /app/mlflow.db /app/mlartifacts     && rm -rf /app/bundle /app/build_store.py

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# Hosts that inject PORT (Render, Hugging Face) are honoured; 8000 otherwise.
CMD ["sh", "-c", "uvicorn ml_platform.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
