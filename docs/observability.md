# Observability

Metrics, dashboards and traces for the two serving tiers. Manifests:
[`k8s/monitoring/`](../k8s/monitoring/), applied with
`kubectl apply -k k8s/monitoring`.

## Architecture

```
  inference-api  ──/metrics──┐                    ┌──► Grafana ──► dashboard
  (application tier)         ├──► Prometheus ─────┘    (provisioned)
  predictor      ──/metrics──┘     (scrapes by
  (model tier)                      annotation)

  both tiers ───────── OTLP/HTTP ──────────────► Jaeger
                       (traceparent propagated
                        across the tier boundary)
```

**No OpenTelemetry Collector.** Jaeger accepts OTLP natively on 4318, which is
what the application exports, so a collector between them would add a hop, a
Deployment and a failure mode while changing nothing about the traces that
arrive. A collector earns its place with many producers to fan in, several
backends to fan out to, or central sampling and redaction. None of that is true
here, and adding it anyway is the kind of technology-for-its-own-sake this
project is meant to argue against.

**No service mesh.** Trace context crosses the tier boundary because both sides
are instrumented in-process: the httpx instrumentation writes `traceparent` on
the outgoing call and the FastAPI instrumentation on the model tier continues
that trace. Both tiers run the same image, so this is one configuration rather
than an integration. A mesh would buy automatic propagation the application
already does for itself.

**Discovery is by annotation.** A pod is scraped because it carries
`prometheus.io/scrape: "true"`, not because Prometheus's config names it. That
matters for the model tier in particular: KServe writes that Deployment, so the
annotation travels on the `InferenceService` and KServe copies it onto the pod.

## What is measured, and what deliberately is not

| Metric | Answers |
|---|---|
| `ml_platform_http_requests_total` | how much traffic, by route and status |
| `ml_platform_http_request_errors_total` | how much of it failed, by class |
| `ml_platform_http_request_duration_seconds` | how slow, as a histogram |
| `ml_platform_prediction_requests_total` | predictions, by outcome |
| `ml_platform_applications_scored_total` / `_flagged_total` | how many applications, and how many crossed the threshold |
| `ml_platform_model_tier_requests_total` | upstream KServe calls: success, error, unavailable, invalid |
| `ml_platform_model_tier_duration_seconds` | upstream latency, measured by the caller |
| `ml_platform_model_ready`, `ml_platform_model_info` | whether it can serve, and which version |

These are the operational signals ADR-003 draws rollback authority from: error
rate, latency, timeouts, serving health. **Accuracy is not here and must not
be.** Realised performance takes years to arrive on this dataset and belongs to a
different clock; putting it on the same dashboard would invite someone to act on
it as though it were operational.

The model tier is timed from the caller's side as well as its own, because only
the caller can see connection setup, queueing and the network — and a timeout is
invisible to the service that never answered.

### Cardinality

A Prometheus series exists per distinct label combination and is held in memory,
so a label carrying a request id, a raw path or an application's values turns one
metric into unbounded many and eventually takes the server down. Every label here
is drawn from a small fixed set:

- **route is the template, never the path.** `/predict`, and on the model tier
  `/v1/models/{name}:predict`. A request matching nothing becomes the constant
  `unmatched`, because the path of a 404 is whatever the caller sent.
- **status class** is four values, not six hundred.
- **outcome** is a fixed vocabulary the code defines.

The one deliberately broad label is the served model version on
`ml_platform_model_info` — attributing a latency or error change to a specific
promoted version is the reason the registry exists, and it is bounded by how
often a model is promoted. `tests/integration/test_observability.py` asserts all
of this, including a guard against the next unbounded label.

No application payload is logged or put on a span. Spans carry batch size, model
version, feature set and served-by, and nothing about the loan.

## Telemetry cannot break serving

A platform whose observability can fail the thing it observes is worse than one
with none. Every metric call is wrapped and swallows its own failure; tracing
setup returns `False` and logs rather than raising; and spans are handed to a
batch processor on a background thread, so an unreachable collector costs a
dropped batch and a log line.

Verified by scaling Jaeger and Prometheus to zero and sending ten predictions:
all ten returned 200 with the correct probabilities, while the exporter logged
`Failed to export span batch due to timeout, max retries or shutdown` on its own
thread.

Locally and under test there is no collector, `OTEL_EXPORTER_OTLP_ENDPOINT` is
unset, and tracing is simply off. That is the default, not a degraded mode.

## Runbook

```bash
kubectl apply -k k8s/monitoring

kubectl -n ml-platform port-forward svc/grafana 3000:3000 &     # dashboard
kubectl -n ml-platform port-forward svc/prometheus 9090:9090 &  # queries, targets
kubectl -n ml-platform port-forward svc/jaeger 16686:16686 &    # traces
```

Grafana opens straight onto **ML platform - serving** with Prometheus already
wired in: the data source and the dashboard are provisioned from ConfigMaps, so a
fresh cluster has them without anyone rebuilding a dashboard from memory.

Grafana runs with anonymous access and no login. It is a laptop cluster, the data
is request counts, and there is no credential to protect — but that is also why
its Service is `ClusterIP`. Exposing it beyond the cluster means turning
anonymous access off first.

## Evidence from real traffic

470 predictions plus deliberate failures, through the Service:

```
requests          /predict 200=453  422=68  503=9   unmatched 404=5
latency /predict  p50 36.8ms   p95 67.4ms   p99 93.5ms
model tier        p50 35.9ms   p95 48.7ms      <- most of the time is the model
applications      906 scored, 0 flagged (both probabilities are far below 0.5)
model tier        success=453  unavailable=9
```

A trace of one `/predict`, both tiers in one trace:

```
inference-api    POST /predict                      dur=38.41ms
  inference-api    POST                             dur=25.78ms   (httpx client)
    model-predictor  POST /v1/models/{name}:predict dur=24.82ms
```

And one failure, with the model tier deleted:

```
[ERROR] inference-api  POST /predict   dur=4020.85ms  otel.status_code=ERROR
[ERROR] inference-api  POST            httpx.ConnectError: Temporary failure in name resolution
```

## Limitations

**Jaeger stores traces in memory.** They are lost when the pod restarts. For a
development cluster that is the honest trade rather than a volume that only
pretends to be durable; a real deployment needs Elasticsearch or Cassandra
behind it.

**Prometheus retains six hours, on an `emptyDir`.** Same reasoning. The
interesting window is the traffic someone just generated.

**Rates need sustained traffic.** A short burst leaves a counter flat by the time
the window closes, and `histogram_quantile` over a zero rate is `NaN` — which
looks like a broken dashboard and is not. The latency figures above come from
four minutes of continuous traffic, not a burst.

**MLflow and the KServe controller are not scraped.** Both expose metrics;
neither carries the annotation yet. The application tiers were the milestone's
subject, and adding targets nobody has a dashboard for would be noise.
