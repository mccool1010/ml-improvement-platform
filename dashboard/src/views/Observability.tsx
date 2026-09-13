import { api } from "../lib/api";
import { count, metric, percent, seconds } from "../lib/format";
import { Caveat, Panel, SectionBody } from "../components/primitives";
import { useSection } from "../lib/useSection";

export function Observability() {
  const obs = useSection(api.observability);
  const health = useSection(api.health);

  return (
    <>
      <div className="view-head">
        <h2>Observability</h2>
        <p>
          Prometheus scrapes the API, Grafana renders it, and OpenTelemetry traces a request from
          the application tier through to KServe. This page is a summary and a set of links —
          Grafana and Jaeger are the observability systems and this does not try to replace
          either.
        </p>
      </div>

      <Panel
        title="Request metrics"
        status={
          obs.data?.available ? "healthy" : obs.loading ? null : "unavailable"
        }
        note="Five-minute windows. A quantile over an empty window is reported as unavailable rather than zero: no traffic is not the same as instant responses."
      >
        <SectionBody
          state={obs}
          rows={2}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <div className="metrics">
              <Cell label="Request rate" value={metric(data.request_rate, 3)} sub="req/s" />
              <Cell label="Error ratio" value={percent(data.error_ratio)} sub="5xx + 4xx / all" />
              <Cell label="p50 /predict" value={seconds(data.latency_p50_seconds)} />
              <Cell label="p95 /predict" value={seconds(data.latency_p95_seconds)} />
              <Cell
                label="p95 model tier"
                value={seconds(data.model_tier_p95_seconds)}
                sub="time inside KServe"
              />
              <Cell
                label="Applications scored"
                value={count(data.applications_scored)}
                sub="counter total"
              />
            </div>
          )}
        </SectionBody>
      </Panel>

      <div className="grid">
        <Panel title="The real tools">
          <SectionBody state={obs} rows={3}>
            {(data) => (
              <ul className="plain">
                <Link
                  label="Grafana"
                  href={data.grafana_url}
                  what="provisioned dashboard: traffic, errors, latency, model tier"
                />
                <Link
                  label="Jaeger"
                  href={data.jaeger_url}
                  what="end-to-end traces, application tier through to the predictor"
                />
                <Link
                  label="Prometheus"
                  href={data.prometheus_url}
                  what="raw metric store and query browser"
                />
              </ul>
            )}
          </SectionBody>
        </Panel>

        <Panel title="Telemetry design rules">
          <ul className="plain">
            <li>
              <strong className="muted">Telemetry failure never fails a prediction.</strong> The
              exporters are best-effort; a dead Prometheus or Jaeger is invisible to{" "}
              <span className="mono">/predict</span>. The{" "}
              <span className="mono">telemetry_failure</span> scenario proves it by scaling both
              to zero.
            </li>
            <li>
              <strong className="muted">No high-cardinality labels.</strong> Route, method,
              status class and serving tier only — never request IDs, model inputs or
              user-supplied values.
            </li>
            <li>
              <strong className="muted">No application payloads in logs or spans.</strong> Loan
              application fields are the data this platform exists to score, not something to
              scatter through a telemetry backend.
            </li>
          </ul>
        </Panel>
      </div>

      {health.data && health.data.status === "degraded" ? (
        <Caveat>
          <strong>The platform is reporting degraded.</strong> That is the expected reading when
          Prometheus or MLflow is unreachable from the API: on the hosted demo, which runs as a
          single container with no monitoring stack, or on a laptop process outside the cluster.
          Predictions are unaffected by design; the components that are down are the ones
          deliberately kept off the request path.
        </Caveat>
      ) : null}
    </>
  );
}

function Cell({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

function Link({
  label,
  href,
  what,
}: {
  label: string;
  href: string | null;
  what: string;
}) {
  return (
    <li>
      {href ? (
        <a href={href} target="_blank" rel="noreferrer">
          {label}
        </a>
      ) : (
        <span className="faint">{label} (no endpoint configured)</span>
      )}{" "}
      — {what}
    </li>
  );
}
