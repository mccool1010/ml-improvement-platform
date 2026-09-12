import { api } from "../lib/api";
import { count, metric, percent, seconds, shortId, when } from "../lib/format";
import { Caveat, KeyValue, Panel, SectionBody, StatusPill } from "../components/primitives";
import { useSection } from "../lib/useSection";

export function Overview() {
  const health = useSection(api.health);
  const model = useSection(api.model);
  const lifecycle = useSection(api.lifecycle);
  const events = useSection(api.events);
  const obs = useSection(api.observability);

  return (
    <>
      <div className="view-head">
        <h2>Overview</h2>
        <p>
          One SBA loan-default model in production, and the machinery that decides whether it
          stays there. Every number on this page is read from the system that owns it — MLflow
          for lineage and the registry, the running API for what is actually loaded, Prometheus
          for traffic. Nothing is cached or reconstructed.
        </p>
      </div>

      <div className="grid-2">
        <Panel
          title="Platform health"
          status={health.data?.status ?? null}
          aside={
            health.data ? (
              <StatusPill
                status={health.data.can_serve ? "healthy" : "unavailable"}
                label={health.data.can_serve ? "can serve" : "cannot serve"}
              />
            ) : null
          }
          note="MLflow or Prometheus being unreachable is degraded, not down: both are kept off the request path, so predictions continue."
        >
          <SectionBody state={health} rows={4}>
            {(data) => (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Component</th>
                      <th>Status</th>
                      <th>Authoritative for</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.components.map((component) => (
                      <tr key={component.name}>
                        <td>{component.name}</td>
                        <td>
                          <StatusPill status={component.status} />
                        </td>
                        <td className="muted">{component.owns ?? "—"}</td>
                        <td className="faint">{component.detail ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </SectionBody>
        </Panel>

        <Panel
          title="Production model"
          aside={
            model.data?.available ? (
              <span className="pill healthy">
                {model.data.name}:{model.data.version}
              </span>
            ) : null
          }
        >
          <SectionBody
            state={model}
            rows={5}
            unavailable={(data) => (data.available ? null : { detail: data.detail })}
          >
            {(data) => (
              <KeyValue
                rows={[
                  ["Registered model", data.name ?? "—"],
                  ["Version / alias", `${data.version ?? "—"} @ ${data.alias ?? "—"}`],
                  ["Feature set", data.feature_set ?? "—"],
                  [
                    "Quality gates",
                    data.gates_passed !== null && data.gates_total !== null ? (
                      <span>
                        {data.gates_passed} of {data.gates_total} passed
                      </span>
                    ) : (
                      "—"
                    ),
                  ],
                  ["Decision threshold", metric(data.decision_threshold, 4)],
                  ["Threshold source", data.threshold_source ?? "—"],
                  ["Served by", data.served_by ?? "—"],
                  ["MLflow run", shortId(data.mlflow_run_id)],
                  ["Git revision", shortId(data.git_revision, 10)],
                  ["Dataset SHA-256", shortId(data.dataset_sha256, 16)],
                ]}
              />
            )}
          </SectionBody>
        </Panel>
      </div>

      <Panel
        title="Serving traffic"
        note="Six numbers from Prometheus, not a dashboard. Grafana and Jaeger remain the observability systems."
      >
        <SectionBody
          state={obs}
          rows={2}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <div className="metrics">
              <Cell label="Request rate" value={metric(data.request_rate, 3)} sub="req/s, 5m avg" />
              <Cell label="Error ratio" value={percent(data.error_ratio)} sub="5m" />
              <Cell label="p50 /predict" value={seconds(data.latency_p50_seconds)} sub="5m" />
              <Cell label="p95 /predict" value={seconds(data.latency_p95_seconds)} sub="5m" />
              <Cell label="p95 model tier" value={seconds(data.model_tier_p95_seconds)} sub="KServe" />
              <Cell label="Applications scored" value={count(data.applications_scored)} sub="total" />
            </div>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Improvement lifecycle"
        note="A stage is filled when this installation has state for it; hollow means the code exists and has not been exercised here."
      >
        <SectionBody state={lifecycle} rows={3}>
          {(stages) => (
            <div className="lifecycle">
              {stages.map((stage) => (
                <div className={`stage ${stage.state ?? "implemented"}`} key={stage.stage}>
                  <div className="name">
                    {stage.state === "observed" ? "● " : "○ "}
                    {stage.stage}
                  </div>
                  <div className="meta">{stage.milestone}</div>
                </div>
              ))}
            </div>
          )}
        </SectionBody>
      </Panel>

      <Panel title="Recent activity" aside={<span className="faint mono">from MLflow runs</span>}>
        <SectionBody state={events} rows={5}>
          {(items) =>
            items.length === 0 ? (
              <p className="muted">No runs are recorded in this tracking store.</p>
            ) : (
              <ul className="events">
                {items.map((event, index) => (
                  <li key={`${event.at}-${index}`}>
                    <span className="at">{when(event.at)}</span>
                    <span className="kind">{event.kind}</span>
                    <span className="summary">{event.summary}</span>
                    <span className="pill">{event.outcome ?? "—"}</span>
                  </li>
                ))}
              </ul>
            )
          }
        </SectionBody>
      </Panel>

      <Caveat>
        <strong>Three clocks, not one.</strong> Input drift is measurable today; operational
        health is measurable in seconds; realised performance needs matured labels, which on
        this dataset take 60 months. Nothing on this dashboard claims to know whether the
        production model has become less accurate.
      </Caveat>
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
