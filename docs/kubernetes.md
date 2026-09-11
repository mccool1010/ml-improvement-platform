# Kubernetes deployment

The inference API from M7, in the image from M8, running as a Kubernetes
workload with its own tracking server.

Manifests: [`k8s/base/`](../k8s/base/). Applied with `kubectl apply -k k8s/base`.

## The problem this milestone had to solve first

M8 found that the local MLflow store records each experiment's artifact root in
its database as an absolute host URI — `file:///C:/mlops/mlartifacts`. Those rows
are data, not configuration. A container could only be made to work by
bind-mounting the artifacts at that literal path, which is a workaround for
exactly one machine. A pod cannot do it at all, and should not try.

So the deployment could not begin with the API. It had to begin with somewhere
for the model to live.

## Architecture

```
  ConfigMap                Service "mlflow"              PVC
  MLFLOW_TRACKING_URI  ──►  ClusterIP :5000  ──►  Deployment mlflow  ──►  2Gi
        │                                         mlflow server
        │                                         --serve-artifacts
        ▼
  Deployment inference-api  ──►  Service "inference-api"  ClusterIP :80
  2 replicas                     (endpoints gated by readiness)
```

**A tracking server, not a shared filesystem.** `mlflow server --serve-artifacts`
makes the server the artifact broker: run artifacts are recorded as
`mlflow-artifacts:/…`, which resolve relative to whichever server the client is
talking to. Clients fetch them over HTTP. No client ever needs the server's disk,
so no path outside the cluster is ever named. That is the whole fix, and it is
why this had to come before the Deployment rather than after it.

**The server runs the project's own image.** Not `ghcr.io/mlflow/mlflow`. The
server and every client then hold the identical MLflow build from `uv.lock`, so
the thing that writes the registry and the thing that reads it cannot drift apart,
and there is no second image to pull or keep current.

**One value comes from Kubernetes.** The ConfigMap carries `MLFLOW_TRACKING_URI`
and nothing else, because that is the only setting that genuinely differs between
a laptop and a cluster. Everything else is in `configs/` inside the image and is
the same everywhere; restating it in a ConfigMap would only give it somewhere to
drift. The value is the Service DNS name `http://mlflow:5000` — that is the
service discovery, and it is why no manifest names a machine or an address.

**No Secret.** There is no credential to hold. The tracking server is reachable
only inside the namespace and has no authentication. When there is one — an
object store's keys, or auth on the server — it belongs in a Secret. An empty one
now would be decoration.

**Model resolution is unchanged.** The pod resolves its model through
`resolve_production()`, by the `production` alias, exactly as the CLI and the
local container do. Kubernetes changed where the registry lives, not how
production is decided. A pod is never handed a model directly and never falls
back to the newest run.

## Two code changes this needed

**`artifact_uri` is now `None` against a server.** `_ensure_experiment` used to
pass an explicit local `artifact_location` on every experiment creation. That was
right for a file store — it enforces the M2 rule that nothing depends on the
working directory — but it is what wrote an absolute Windows path into the
database for every client that came later. Against an `http://` tracking URI no
location is passed, and the server assigns its own.

**`/ready` answers 503 when no model is loaded.** It previously answered 200 with
`{"status": "not_ready"}` in the body. A Kubernetes readiness probe decides from
the status code alone and never reads the body, so a 200 there would have put a
pod holding no model into the Service's endpoints — the precise thing readiness
exists to prevent. The body is unchanged; only the code is new. `/health` still
answers 200 regardless, because a pod with no promoted model is alive and must
not be restarted in a loop over it.

## Health management

| Probe | Endpoint | Question it asks |
|---|---|---|
| `startupProbe` | `/health` | Has it finished loading? Up to 150s, so a slow artifact download is not counted against liveness |
| `livenessProbe` | `/health` | Is the process serving? Never `/ready` — no model is not a reason to restart |
| `readinessProbe` | `/ready` | Can it actually score? 503 removes the pod from the Service |

That split is the whole point of having two endpoints. A pod with no promoted
model stays **running and out of rotation**, which is visible as `1/2 Running` —
not `CrashLoopBackOff`, and not quietly serving errors.

## Runbook

Prerequisites: Docker, and a local cluster. Docker Desktop's Kubernetes
(Settings → Kubernetes → Enable) is what this was verified on.

```bash
# 1. Build the image the cluster will run.
docker build -f docker/Dockerfile -t ml-platform-api:m10 .

# 2. Make it visible to the cluster. Docker Desktop runs Kubernetes as a kind
#    node with its own image store, so a locally built image has to be imported.
docker save ml-platform-api:m10 -o /tmp/api.tar
docker cp /tmp/api.tar desktop-control-plane:/api.tar
docker exec desktop-control-plane ctr -n k8s.io images import /api.tar

# 3. Apply everything.
kubectl apply -k k8s/base

# 4. Wait for the tracking server.
kubectl -n ml-platform rollout status deploy/mlflow

# 5. The API pods are Running but NOT ready: the registry is empty. This is
#    correct, and is worth looking at once before fixing it.
kubectl -n ml-platform get pods

# 6. Copy the promoted model into the cluster's registry.
kubectl -n ml-platform port-forward svc/mlflow 5000:5000 &
uv run python scripts/seed_registry.py --destination http://localhost:5000

# 7. Restart the API. The model is loaded once at startup, by design (M7), so a
#    pod that started against an empty registry never retries -- it stays
#    correctly unready until it is replaced.
kubectl -n ml-platform rollout restart deploy/inference-api
kubectl -n ml-platform rollout status deploy/inference-api

# 8. Reach the service.
kubectl -n ml-platform port-forward svc/inference-api 8080:80 &
curl -s localhost:8080/health
curl -s localhost:8080/ready
```

### Seeding is a transfer, not a promotion

[`scripts/seed_registry.py`](../scripts/seed_registry.py) copies the version that
already carries the `production` alias into the destination registry, with its
version tags verbatim — platform run id, git revision, dataset checksum, gate
counts. It evaluates no gate and decides nothing; it refuses to run if the source
has no production alias. The model that M6 accepted is the model that gets
served, and the registered version in the cluster still records the decision that
put it there.

The alternative — re-running `promote` against the cluster — would mint a
*different* version and re-litigate a decision that was already made. It would
also fail, correctly: the reproducibility gate requires a clean revision, and
deploying is not a moment when the working tree is clean.

## Three things running it found

Everything below was found by deploying, not by reading the manifests.

**The tracking server was OOMKilled.** `mlflow server` defaults to four worker
processes, each importing the whole of MLflow, and four of them exceeded a 1Gi
limit before the server answered a single probe. It now runs `--workers=1`, which
is also all a SQLite backend on a ReadWriteOnce volume can support, with 2Gi.

**MLflow 3 rejects Service DNS names by default.** `--allowed-hosts` defaults to
localhost and private IP literals, so a request whose `Host` header is
`mlflow:5000` is refused as a DNS-rebinding attempt. Every in-cluster client
sends exactly that. The Service names are now listed explicitly.

**An unreachable tracking server hung the API in startup.** The MLflow client
retries a connection failure with exponential backoff, and that retry runs inside
the API's startup lifespan -- so while the server was crash-looping, the pods sat
in `Waiting for application startup` serving nothing at all, rather than starting
and reporting themselves unready. This is the same shape as the M8 finding about
`/app` ownership: the failure path was designed correctly and then blocked by a
library's retry loop. `MLFLOW_HTTP_REQUEST_MAX_RETRIES` and
`MLFLOW_HTTP_REQUEST_TIMEOUT` in the ConfigMap bound the attempt, and the M7
contract holds again.

## What is still a limitation

**The cluster's registry is a copy, not the source of truth.** There are now two
registries: the local store the CLI writes to, and the cluster's. A promotion on
the laptop does not appear in the cluster until it is seeded again. The honest
fix is for the tracking server to be *the* registry — for `promote` to write to it
directly — which means the server outliving the cluster and being reachable from
wherever training runs. That is a real infrastructure decision, not a manifest
change.

**SQLite on one ReadWriteOnce volume.** One writer, one node, `Recreate` rollout.
It is genuinely fine for a single-node local cluster and genuinely will not scale
past one: a multi-node cluster needs a real database and object storage behind
the server. Nothing in the application would change — only the server's flags.

**Readiness does not recover without a restart.** The model is loaded once at
startup, which is the M7 design and the reason a request never pays for a
registry lookup. The cost is that a pod which started against an empty or
unreachable registry stays unready until it is replaced. In Kubernetes that is a
`rollout restart`, and a pod in that state is already out of the Service's
endpoints, so nothing is served wrongly in the meantime. Making the load
retryable would be a change to the serving design, not to a manifest.

**The served model reports `threshold_source: "fallback"`.** Registered v1
predates the `validation_threshold_at_capacity` tag, so the API decides at 0.5
rather than the review-capacity operating point, and says so in every response.
Carried over from M7, unchanged here.
