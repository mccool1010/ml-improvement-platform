import { api } from "../lib/api";
import { metric, when } from "../lib/format";
import { Caveat, KeyValue, Panel, SectionBody } from "../components/primitives";
import { useSection } from "../lib/useSection";

export function Drift() {
  const drift = useSection(api.drift);

  return (
    <>
      <div className="view-head">
        <h2>Drift &amp; retraining</h2>
        <p>
          Population Stability Index per feature against the training reference, computed over a
          window of the production stream. Drift can trigger a retraining run; it can never
          promote, reject or roll back anything on its own.
        </p>
      </div>

      <Panel
        title="Latest drift check"
        status={
          drift.data?.available
            ? drift.data.decision === "retrain"
              ? "degraded"
              : "healthy"
            : null
        }
        aside={
          drift.data?.available && drift.data.decision ? (
            <span className={`pill ${drift.data.decision === "retrain" ? "degraded" : "healthy"}`}>
              {drift.data.decision}
            </span>
          ) : null
        }
      >
        <SectionBody
          state={drift}
          rows={5}
          command="python -m ml_platform drift"
          hint="These commands record to the MLflow store they are pointed at. To see the result on a cluster dashboard, set MLFLOW_TRACKING_URI to the cluster's MLflow (port-forward it to localhost:5000) before running them."
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <>
              <KeyValue
                rows={[
                  ["Decision", data.decision ?? "—"],
                  [
                    "Window",
                    data.window_start && data.window_end
                      ? `${data.window_start} .. ${data.window_end}`
                      : "—",
                  ],
                  ["Scenario", data.scenario ?? "—"],
                  [
                    "Features over threshold",
                    data.n_drifted !== null && data.n_features !== null
                      ? `${data.n_drifted} of ${data.n_features}`
                      : "—",
                  ],
                  ["Max PSI", metric(data.max_psi, 4)],
                  [
                    "Trigger rule",
                    data.threshold_psi !== null && data.min_drifted_features !== null
                      ? `PSI ≥ ${metric(data.threshold_psi, 2)} on ≥ ${data.min_drifted_features} feature(s)`
                      : "—",
                  ],
                  ["Checked at", when(data.checked_at)],
                ]}
              />
              {data.top_features.length > 0 ? (
                <div className="table-scroll" style={{ marginTop: 18 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th className="num">PSI</th>
                        <th style={{ width: "45%" }}>Against threshold</th>
                        <th>Drifted</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.top_features.map((feature) => {
                        const limit = Math.max(data.max_psi ?? feature.psi, 0.001);
                        return (
                          <tr key={feature.feature}>
                            <td>{feature.feature}</td>
                            <td className="num">{metric(feature.psi, 4)}</td>
                            <td>
                              <div className={`bar ${feature.drifted ? "over" : ""}`}>
                                <span
                                  style={{
                                    width: `${Math.min(100, (feature.psi / limit) * 100)}%`,
                                  }}
                                />
                              </div>
                            </td>
                            <td>
                              <span className={`pill ${feature.drifted ? "degraded" : "healthy"}`}>
                                {feature.drifted ? "yes" : "no"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Retraining response"
        note="A retrained candidate is offered to the same seven gates as any other candidate. Drift is the reason it was trained, never the reason it is accepted."
      >
        <SectionBody
          state={drift}
          rows={3}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => {
            const retraining = data.retraining as Record<string, unknown>;
            if (Object.keys(retraining).length === 0 || retraining.ran === false) {
              return (
                <p className="muted">
                  No retraining run is linked to this drift event.
                  {data.decision === "retrain" ? (
                    <>
                      {" "}
                      The check recommended one; run{" "}
                      <span className="mono">python -m ml_platform retrain</span> to produce a
                      candidate.
                    </>
                  ) : null}
                </p>
              );
            }
            return (
              <KeyValue
                rows={[
                  ["Ran", "yes"],
                  ["Candidate run", String(retraining.candidate_run_id ?? "—")],
                  [
                    "Candidate validation AP",
                    metric(retraining.validation_average_precision as number | null, 4),
                  ],
                  ["Retraining window rows", String(retraining.window_rows ?? "—")],
                  [
                    "Outcome",
                    <span className={`pill ${retraining.promoted ? "healthy" : "degraded"}`}>
                      {retraining.promoted ? "promoted" : "not promoted"}
                    </span>,
                  ],
                ]}
              />
            );
          }}
        </SectionBody>
      </Panel>

      <Caveat>
        <strong>Drift is not performance.</strong>{" "}
        {drift.data?.performance_note ??
          "Realised performance needs matured labels, which take 60 months on this dataset."}{" "}
        This view reports that the inputs moved. It does not, and cannot, report that the model
        became less accurate — and no ground truth is invented to close the gap.
      </Caveat>
    </>
  );
}
