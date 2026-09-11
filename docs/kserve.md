# KServe model serving

The model tier. [ADR-002](decisions/ADR-002-serving.md) decided at M0 that KServe
would be the canonical serving path and FastAPI the application layer in front of
it; this is that, implemented.

Manifests: [`k8s/kserve/`](../k8s/kserve/). Applied with `kubectl apply -k k8s/kserve`.

## The two tiers, and what each owns

```
  client ──► Service inference-api ──► FastAPI (2 replicas)
                                        │  validates the application
                                        │  resolves `production` from MLflow
                                        │  reports version + threshold
                                        ▼
                          Service sba-loan-default-predictor
                                        │
                                        ▼
                       InferenceService sba-loan-default (KServe)
                          builds features, scores, returns probabilities
                                        │
                                        ▼
                          PVC mlflow-store  ← written by the tracking server
```

| | Application tier | Model tier |
|---|---|---|
| Runs | `ml_platform.api` | `ml_platform.serving.predictor`, under KServe |
| Owns | request validation, domain errors, the `production` alias lookup, version metadata, the decision threshold and `flagged` | the artifact, feature building, `predict_proba` |
| Holds the model | **no** — in Kubernetes | yes |
| Managed by | a plain Deployment | the KServe controller |

**There is one production serving path.** The plain Deployment does not serve the
model in the cluster: `ML_PLATFORM_PREDICTOR_URL` in the ConfigMap is what turns
it into the application tier, and its presence is the only switch. Unset — on a
laptop, and in the plain container from M8 — the API loads the model itself,
because there is no InferenceService to call. That difference is the one ADR-002
predicted, and `tests/integration/test_serving.py` covers the seam.

**Feature building belongs to the model tier.** The application tier sends
register-shaped records, not features. Keeping the transformation and the
pipeline in one tier is what stops the served feature representation drifting
from the trained one across a deployment.

## Environment

| | |
|---|---|
| KServe | v0.20.0, `kserve/kserve-controller:v0.20.0` |
| Mode | RawDeployment — **no Knative, no Istio, no Gateway API** |
| cert-manager | v1.17.0 (required for KServe's webhooks) |
| Kubernetes | v1.36.1, Docker Desktop single node |
| Protocol | KServe V1 — `POST /v1/models/<name>:predict` |

Install:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.0/cert-manager.yaml
kubectl -n cert-manager rollout status deploy/cert-manager-webhook

kubectl create namespace kserve
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.20.0/kserve-crds.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.20.0/kserve.yaml
```

Then switch the controller to raw mode. KServe defaults to Serverless, which
waits on Knative forever in a cluster that has none:

```bash
kubectl -n kserve patch cm inferenceservice-config --type merge \
  -p '{"data":{"deploy":"{\n  \"defaultDeploymentMode\": \"RawDeployment\"\n}"}}'
# also set ingress.disableIngressCreation=true; keep the ingressGateway keys, the
# controller refuses to start without them even when it creates no ingress.
kubectl -n kserve rollout restart deploy/kserve-controller-manager
```

KServe's own "standard mode" installer additionally pulls in Istio 1.27.1 and
Gateway API. Neither is installed here: nothing outside the cluster reaches the
model tier, and the application tier reaches it by cluster DNS. See the debt
section for what that costs later.

## Why not the stock MLflow runtime

KServe ships `kserve-mlserver`, built on `seldonio/mlserver:1.7.1`, which claims
the `mlflow` model format. It is not used, for two reasons found by running it.

**Version parity.** The registered artifact's own `requirements.txt` pins
`scikit-learn==1.9.0`, `cloudpickle==3.1.2`, `mlflow==3.16.0`, `numpy==2.5.2`.
The pipeline is cloudpickle-serialised, and unpickling a scikit-learn estimator
under a different scikit-learn version is explicitly unsupported. A runtime image
with its own pinned versions cannot be trusted to reproduce this model's
predictions, and reproducing them exactly is this project's whole contract. The
`ServingRuntime` therefore runs the project's own image, whose environment comes
from the `uv.lock` that produced the artifact — the same reasoning that put the
tracking server on that image in M10.

**Probabilities, not labels.** The model was logged with the sklearn flavor's
default, so its `python_function` flavor records `predict_fn: predict`. Served
through `mlflow models serve`, it answers `{"predictions": [0, 0]}` — hard
labels. This project decides by comparing a probability to a review-capacity
threshold, so a label throws away the number the decision is made on. The fix is
not to re-log the model with `pyfunc_predict_fn`: that would mint a new registry
version and re-open a promotion decision M6 already made. Instead the predictor
loads the **sklearn flavor** and calls `predict_proba`, which is what every
serving path in this project has always done.

## How the model reaches the pod

`storageUri` is a `pvc://` reference to the volume the tracking server writes to,
at the path the registry records for the promoted version:

```
registry source   mlflow-artifacts:/1/<run id>/artifacts/model
storageUri        pvc://mlflow-store/artifacts/1/<run id>/artifacts/model
```

KServe mounts the claim read-only at `/mnt/models` with that `subPath` — no
storage initializer, no copy, and nothing outside the cluster is named.

The run id is not typed by hand.
[`scripts/kserve_model_uri.py`](../scripts/kserve_model_uri.py) reads the
`production` alias — the same mechanism M6 uses to decide what production is —
and translates the registry's recorded source into the volume path:

```bash
kubectl -n ml-platform port-forward svc/mlflow 5000:5000 &
uv run python scripts/kserve_model_uri.py --tracking-uri http://localhost:5000
```

It prints; it applies nothing. Moving the model tier onto a newly promoted
version is a deliberate act: derive the URI, edit the `storageUri` and the
`ml-platform.io/*` annotations, re-apply. **Nothing watches the registry.** M10
established that the cluster registry is seeded rather than synchronised, and a
reconciler here would be a different milestone with its own failure modes.

## Health and readiness

The model tier answers `/health` with 503 until the pipeline is deserialised, so
an unready predictor stays out of its Service.

The application tier's readiness is **not** a snapshot taken at startup. In
Kubernetes it holds no model at all — only the registry metadata it resolved once
and a dependency — so `/ready` asks the model tier live on every probe. A
snapshot would leave it reporting ready after the InferenceService had gone, or
unready forever because it happened to start first. Checking live is what makes
the two Deployments independent of their start order, which was worth having:
deleting and re-applying the InferenceService brought both tiers back with no
restart of the API.

Loading the model once at startup is unchanged where it still applies — the
in-process path, and the registry metadata lookup.

## Runbook

Assumes `k8s/base` is deployed and the registry seeded (see
[docs/kubernetes.md](kubernetes.md)).

```bash
# 1. Build and import the image. Both tiers run it.
docker build -f docker/Dockerfile -t ml-platform-api:m11 .
docker save ml-platform-api:m11 -o /tmp/api.tar
docker cp /tmp/api.tar desktop-control-plane:/api.tar
docker exec desktop-control-plane ctr -n k8s.io images import /api.tar

# 2. Apply the model tier.
kubectl apply -k k8s/kserve
kubectl -n ml-platform get isvc sba-loan-default

# 3. Point the application tier at it (already in k8s/base) and apply.
kubectl apply -k k8s/base

# 4. Score something.
kubectl -n ml-platform port-forward svc/inference-api 8080:80 &
curl -s localhost:8080/ready          # served_by: kserve
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d @request.json
```

The image tag is mutable and `imagePullPolicy` is `IfNotPresent`, so re-importing
the same tag does **not** restart anything. Delete the pods to pick it up —
`kubectl rollout restart` does not work on the predictor, because the KServe
controller owns that Deployment and reconciles the annotation away:

```bash
kubectl -n ml-platform delete pod -l serving.kserve.io/inferenceservice=sba-loan-default
```

## Architectural debt

**Raw mode has no traffic splitting.** Canary rollout is M14's requirement, and
`canaryTrafficPercent` is a Serverless-mode feature; raw mode gives a plain
Deployment and Service. M14 will need either Knative, or a Gateway API
implementation plus KServe's ingress layer, or percentage splitting done at the
application tier. That is a real cost of keeping this cluster small, and it is
better recorded now than discovered then. ADR-002's note that "KServe requires
Knative and a service mesh" is accurate for the canary capability and premature
for this milestone.

**The InferenceService pins a run id, by hand.** Promotion moves an alias; the
manifest does not follow. The derivation script closes the gap between "what the
registry says" and "what the manifest claims", but someone still edits and
re-applies. Automating it is a deliberate later decision, not an oversight.

**The predictor speaks V1, not the Open Inference Protocol.** V2 is tensor-shaped
and this model's features are named, mixed-type columns. V1 keeps the records
legible; the cost is that generic V2 tooling will not talk to it directly.

**One replica, no autoscaling.** `minReplicas: maxReplicas: 1`. Sizing the model
tier is not this milestone's question, and `minReplicas: 0` is not honoured in
raw mode without KEDA.
