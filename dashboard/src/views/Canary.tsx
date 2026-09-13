import { api } from "../lib/api";
import { metric, when } from "../lib/format";
import { Caveat, KeyValue, Panel, SectionBody } from "../components/primitives";
import { useSection } from "../lib/useSection";

export function Canary() {
  const canary = useSection(api.canary);

  return (
    <>
      <div className="view-head">
        <h2>Canary &amp; rollback</h2>
        <p>
          KServe runs in RawDeployment mode, which has no traffic-splitting primitive, so the
          split happens in the application tier: a SHA-256 hash of the routing key into 10,000
          buckets. The same caller lands on the same tier every time, which is what makes the
          comparison mean anything.
        </p>
      </div>

      <Panel
        title="Traffic allocation"
        status={canary.data?.active ? "degraded" : "healthy"}
        aside={
          canary.data ? (
            <span className={`pill ${canary.data.active ? "degraded" : "healthy"}`}>
              {canary.data.active ? "canary active" : "no canary"}
            </span>
          ) : null
        }
      >
        <SectionBody
          state={canary}
          rows={3}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <>
              <div className="split" style={{ marginBottom: 16 }}>
                <span className="muted" style={{ minWidth: 78 }}>
                  canary
                </span>
                <div className="bar">
                  <span style={{ width: `${Math.min(100, data.traffic_percent)}%` }} />
                </div>
                <span className="n">{data.traffic_percent.toFixed(1)}%</span>
              </div>
              <KeyValue
                rows={[
                  ["Incumbent version", data.incumbent_version ?? "—"],
                  ["Candidate version", data.candidate_version ?? "—"],
                  ["Allocation", `${data.traffic_percent.toFixed(1)}% to the candidate`],
                  ["Routing", "sha256(routing key) mod 10000, sticky per key"],
                  ["Traffic cap", "99% — a canary can never be given all traffic"],
                ]}
              />
            </>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Last canary decision"
        note="Decisions are recorded as MLflow runs, so the reason a rollback happened survives the process that decided it."
      >
        <SectionBody
          state={canary}
          rows={3}
          command="python -m ml_platform canary --action evaluate"
          hint="These commands record to the MLflow store they are pointed at. To see the result on a cluster dashboard, set MLFLOW_TRACKING_URI to the cluster's MLflow (port-forward it to localhost:5000) before running them."
          unavailable={(data) =>
            data.available && data.last_decision === null
              ? { detail: data.detail ?? "no canary decision has been recorded yet" }
              : data.available
                ? null
                : { detail: data.detail }
          }
        >
          {(data) => (
            <>
              <KeyValue
                rows={[
                  [
                    "Decision",
                    <span
                      className={`pill ${
                        data.last_decision === "rollback" ? "unavailable" : "healthy"
                      }`}
                    >
                      {data.last_decision ?? "—"}
                    </span>,
                  ],
                  ["Reason", data.last_reason ?? "—"],
                  ["Decided at", when(data.decided_at)],
                ]}
              />
              {Object.keys(data.signals).length > 0 ? (
                <div className="table-scroll" style={{ marginTop: 16 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Signal</th>
                        <th className="num">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.signals)
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(([key, value]) => (
                          <tr key={key}>
                            <td>{key}</td>
                            <td className="num">{metric(value, 4)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          )}
        </SectionBody>
      </Panel>

      <Panel title="Rollback signals">
        <SectionBody
          state={canary}
          rows={3}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <ul className="plain">
              {data.rollback_signals.map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
            </ul>
          )}
        </SectionBody>
      </Panel>

      <Caveat>
        <strong>Accuracy is not a rollback signal.</strong>{" "}
        {canary.data?.signal_note ??
          "Rollback uses operational signals only: error rate, latency, upstream failures and serving health."}{" "}
        Nor is a canary ever promoted on these signals: promotion still goes through the quality
        gates.
      </Caveat>
    </>
  );
}
