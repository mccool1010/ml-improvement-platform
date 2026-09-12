# Canary and rollback

Putting a gated candidate in front of a share of real traffic before it becomes
production, and getting it back out quickly when it misbehaves.

```
M6 gates pass ──► register under the CANARY alias ──► route N% of traffic
                        (production unchanged)              │
                                                            ▼
                                              evaluate operational signals
                                                   │                │
                                            promote │                │ rollback
                                                    ▼                ▼
                                    production alias moves    traffic returns,
                                                              alias untouched
```

The safety property: **the production alias moves in exactly one function**,
`canary_pipeline.complete`, which refuses unless all three hold — the decision
says `promote`, a canary is still running, and it is running *this* candidate.
The last two exist because a `CanaryDecision` is a durable object: replaying an
old one after the canary ended would move the alias again, and if production had
advanced it would move it backwards. There is no path from "a candidate exists"
to "the candidate is production" that skips either the M6 gates or the canary.

Two configuration guards back this up. The canary alias may not equal the
production alias — that would make starting a canary move production
immediately — and an allocation of 100% is refused, because it leaves the
incumbent with no traffic, nothing to compare against and nothing already
serving to fall back to. The cap is 99%.

## Routing, and why it lives in the application tier

KServe runs here in **RawDeployment** mode. That mode creates a plain Deployment
and Service per InferenceService and has no traffic-splitting primitive —
`canaryTrafficPercent` is Serverless-only, a gap recorded as debt at M11. The
three ways to close it:

| Option | Cost |
|---|---|
| Knative | a large dependency, and the reason RawDeployment was chosen was to avoid it |
| A service mesh | larger still, for one percentage |
| Split in the tier that already fronts the model | none — it already exists |

ADR-002 already makes FastAPI the application tier in front of KServe, so the
split happens there. **KServe still serves every model**: the canary is a second
InferenceService, not a second way of serving. No mesh was introduced.

### Deterministic, sticky routing

A percentage implemented with `random()` cannot be tested, cannot be reproduced
from a report, and sends the same caller to different models on consecutive
identical requests. Instead:

```
bucket = sha256(routing_key)[:8] % 10000
tier   = canary if bucket < traffic_percent * 100 else production
```

The routing key is an explicit `x-ml-routing-key` header if present, otherwise a
digest of the request payload. Consequences worth having: the same application
always reaches the same tier, a test asserts an exact split rather than a
statistical one, and a decision report can name which requests it covered.

SHA-256 rather than `hash()`, which is randomised per process and would route the
same key differently in each API replica.

### Where the allocation lives

The canary's identity comes from the registry (`canary` alias) and its traffic
share from configuration — `ML_PLATFORM_CANARY_URL` and
`ML_PLATFORM_CANARY_TRAFFIC_PERCENT`. Both replicas therefore agree by
construction rather than by coordination. The cost is that changing the
percentage is a config change, not a live dial; see Limitations.

## Decision signals

Every signal is **operational**, observable in minutes, and needs no labels.

| Check | Default | What it catches |
|---|---|---|
| `candidate_healthy` | must be healthy | a tier that cannot serve at all — decided first, and alone |
| `sufficient_traffic` | ≥ 50 requests | too little evidence; a **hold**, never a failure |
| `error_rate_absolute` | ≤ 2% | unacceptable however the incumbent is doing |
| `error_rate_vs_incumbent` | ≤ 2× | worse than what it is replacing, inside the ceiling |
| `upstream_failure_rate` | ≤ 1% | timeouts, which a status-code count hides |
| `latency_p95_absolute` | ≤ 1s | slow in absolute terms |
| `latency_vs_incumbent` | ≤ 1.5× | slower than what it is replacing |

Two comparisons per signal, because each catches what the other misses: an
absolute ceiling alone passes a candidate twice as slow as production, and a
relative check alone passes one that matches a badly behaved incumbent.

**Accuracy is not a signal, and a test asserts it isn't.** ADR-003 gives rollback
authority to error rate, latency, timeouts and serving health only. Realised
performance takes five years to arrive on this dataset, so a canary waiting for
it would never conclude, and one using an *estimate* would act on a number nobody
can check. Offline quality was already judged by the M6 gates before any traffic
moved. What a canary adds is the question those gates cannot answer: does this
model behave itself when real requests hit it.

So the decision is narrow and worth stating: **the candidate is not visibly worse
to operate than the incumbent.** Not "better".

## Rollback: two paths, and what each one actually does

This distinction matters, and the first version of this document got it wrong.

**In process.** `CanaryRouter.stop()` is a single assignment. The next request
routes to the incumbent — no restart, no registry write, nothing to drain. This
is the path a rollback triggered inside the API takes, and the one the tests
exercise.

**From the command line.** `python -m ml_platform canary --action rollback` runs
in its *own* process. The router it builds dies with it, so stopping that router
would roll back nothing at all: the API replicas serving real traffic never see
it. The CLI therefore passes a `KubernetesTrafficController`, which

1. patches `ML_PLATFORM_CANARY_TRAFFIC_PERCENT` to `0` in the ConfigMap, and
2. restarts `deployment/inference-api` and waits for the rollout,

because a ConfigMap change alone is invisible to a pod that already read its
allocation at startup. Both steps are required; the report names which mechanism
was used, and `--no-apply` makes it an explicit dry run that says so.

Neither path touches the registry. The production alias never moved, so there is
nothing to restore.

## Completing does not reload a running predictor

`complete()` moves the production alias in the registry. It does **not** change
what the cluster is serving. The KServe predictor loaded its artifact from the
`storageUri` pinned in `k8s/kserve/31-inferenceservice.yaml` at startup, and the
API resolved its model once at startup too — both M10/M11 limitations, unchanged
here.

So after a successful canary the sequence is: alias moves, traffic returns to the
production tier, and **the production tier is still serving the old model** until
its InferenceService is re-pointed at the new version and the API restarted.
Deriving the new `storageUri` is what `scripts/kserve_model_uri.py` is for. This
is stated rather than hidden because the alternative — implying the alias move is
a deployment — is exactly the sort of claim this project exists to avoid.

## A failing canary never hides behind production

`predict_with` has no fallback. When the canary tier fails, the request fails —
503 — rather than being quietly re-served by the incumbent. Falling back would
make the error rate the decision rests on read as zero, and the canary would pass
on the strength of requests the candidate never served. That is asserted by a
test, and visible in the rollback demonstration below as 92 real 503s.

## Where measurements come from

`SignalSource` has two implementations. `InProcessSignals` reads this process's
own counters, which is what makes the decision testable with no cluster.
`PrometheusSignals` queries the M12 Prometheus, which is what makes it *correct*
in the cluster: with two API replicas, in-process counters see only one replica's
share, and a canary judged on half the evidence is judged wrongly. The report
records which source was used.

Canary metrics are **new series** — `ml_platform_canary_*` — not a `tier` label
added to the M12 metrics. Adding a label changes every series a metric produces
and would break the M12 dashboard queries. Nothing in M12 sees any difference.

## Demonstrated end to end

Against a real MLflow registry (an isolated store, so the cluster is untouched),
400 requests at a 30% allocation:

**Success**
```
canary start   canary alias -> v2   (production still v1)
traffic        400 requests, HTTP {200: 400}; production 276, canary 124
  [PASS] candidate_healthy        healthy
  [PASS] sufficient_traffic       124 >= 50
  [PASS] error_rate_absolute      0.0 <= 0.02
  [PASS] error_rate_vs_incumbent  1.0 <= 2.0
  [PASS] upstream_failure_rate    0.0 <= 0.01
  [PASS] latency_p95_absolute     0.001 <= 1.0
  [PASS] latency_vs_incumbent     1.0 <= 1.5
decision       PROMOTE
registry       production -> v2, canary alias cleared
```

**Rollback, unreachable candidate**
```
traffic        400 requests, HTTP {200: 308, 503: 92}   <- failures surfaced, not masked
  [FAIL] candidate_healthy        unhealthy
decision       ROLLBACK
registry       production -> v1   (unchanged)
```

**Live CLI rollback, against the cluster** — two real InferenceServices, 40%
allocation:
```
before   HTTP {200: 150}   production 87, canary 63   -> canary share 42.0%
         python -m ml_platform canary --action rollback
         set ML_PLATFORM_CANARY_TRAFFIC_PERCENT=0.0 in configmap/inference-api-config
         deployment/inference-api rolled and waited on
after    HTTP {200: 150}   production 150, canary 0   -> canary share 0.0%
         production alias unchanged; the canary InferenceService is still Ready
```

**Rollback, healthy tier with an elevated error rate**
```
  [FAIL] error_rate_absolute      0.195652 > 0.02
  [FAIL] error_rate_vs_incumbent  inf > 2.0
decision       ROLLBACK
registry       production -> v1   (unchanged)
```

## How the alias behaves

M6's `register_candidate` registers a version *and* moves the production alias in
one step, so a gated candidate was production before any traffic reached it. M14
adds an opt-in `alias` argument: `None` keeps the existing behaviour exactly, so
`python -m ml_platform promote` is unchanged, and the canary path passes the
canary alias instead. The gate report is still handed to `register_candidate`,
which refuses a candidate whose gates failed — the canary is an extra hurdle,
never a way round M6.

On rollback the registry is **not touched at all**. There is nothing to restore,
because the production alias never moved, so there is no window in which
production is ambiguous.

## Limitations

- **The traffic percentage is not a live dial**, and a rollback costs a
  rollout. It comes from configuration so replicas agree; changing it means
  patching the ConfigMap and restarting the Deployment, which is what the
  controller does. A true live dial needs shared state — a control endpoint plus
  a store — and is a deliberate later decision. In practice a CLI rollback takes
  as long as a rolling restart of the API, tens of seconds here, not the
  milliseconds an in-process stop takes.
- **Automatic rollback is not wired to an alerting loop.** `canary --action
  decide` evaluates and acts when run; nothing runs it on a timer. Scheduling is
  the same deployment concern deferred at M13.
- **Progressive traffic ramps are not implemented.** The allocation is one
  number, not a 1% → 10% → 50% schedule.
- **The second InferenceService is not templated.** Deploying a canary model tier
  means copying `k8s/kserve/31-inferenceservice.yaml` with a new name and
  `storageUri`. A kustomize overlay would generate it; that is deferred.
- **Two kinds of demonstration, and they prove different things.** The
  promote/rollback registry flows were run against a real MLflow registry with
  *stub* model tiers, because those flows are about aliases and decisions. The
  live CLI rollback was run against the cluster with **two real KServe
  InferenceServices** — `sba-loan-default` and `sba-loan-default-canary` — and
  measured 42% of 150 requests reaching the canary before, 0% after. Both tiers
  serve the same artifact, because the registry holds one model version, so the
  live run proves routing, health, metrics and rollback and proves nothing about
  a quality difference between two models.
- **The CLI reports the allocation from local configuration, not the cluster's.**
  A rollback run from a laptop whose `configs/base.yaml` says 0% will print
  `0.0% traffic` in its decision report even though the cluster was at 40%. The
  rollback itself is correct — it patches the ConfigMap regardless — but the
  number in the report is the local view. Reading it back from the ConfigMap
  first would fix it.
