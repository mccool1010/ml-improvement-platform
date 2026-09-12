import { api } from "../lib/api";
import { METRIC_LABELS, metric, shortId, when } from "../lib/format";
import { Caveat, KeyValue, MetricGrid, Panel, SectionBody } from "../components/primitives";
import { useSection } from "../lib/useSection";

/** The first three headline metrics, shown inline in the version table. */
const TABLE_METRICS = Object.keys(METRIC_LABELS).slice(0, 3);

export function Models() {
  const model = useSection(api.model);
  const promotions = useSection(api.promotions);
  const gates = useSection(api.gates);

  return (
    <>
      <div className="view-head">
        <h2>Models &amp; promotion</h2>
        <p>
          The registry is the only thing that decides what production is: the alias{" "}
          <span className="mono">production</span> on a registered version. A candidate reaches
          it by passing every blocking quality gate, and by no other route — there is no manual
          override in the promotion path.
        </p>
      </div>

      <Panel title="Current production version">
        <SectionBody
          state={model}
          rows={5}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <>
              <KeyValue
                rows={[
                  ["Model", `${data.name ?? "—"} v${data.version ?? "—"}`],
                  ["Alias", data.alias ?? "—"],
                  ["Decision split", data.decision_split ?? "—"],
                  [
                    "Gates",
                    data.gates_passed !== null && data.gates_total !== null
                      ? `${data.gates_passed} / ${data.gates_total} passed`
                      : "—",
                  ],
                  ["Platform run", shortId(data.platform_run_id, 18)],
                  ["MLflow run", shortId(data.mlflow_run_id, 18)],
                  ["Git revision", shortId(data.git_revision, 10)],
                  ["Dataset SHA-256", shortId(data.dataset_sha256, 16)],
                ]}
              />
              <div style={{ marginTop: 16 }}>
                <MetricGrid metrics={data.metrics} />
              </div>
            </>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Registered versions"
        aside={
          promotions.data?.available ? (
            <span className="faint mono">
              production: v{promotions.data.production_version ?? "none"}
            </span>
          ) : null
        }
        note={
          promotions.data?.available &&
          promotions.data.unregistered_candidates !== null ? (
            <>
              <strong className="mono">{promotions.data.unregistered_candidates}</strong> training
              run(s) in this store were never registered. That is the honest denominator: a
              registry on its own makes it look as though everything ever trained was promoted.
            </>
          ) : null
        }
      >
        <SectionBody
          state={promotions}
          rows={4}
          command="python -m ml_platform promote"
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Aliases</th>
                    <th>Gates</th>
                    {TABLE_METRICS.map((key) => (
                      <th className="num" key={key}>
                        {METRIC_LABELS[key]}
                      </th>
                    ))}
                    <th>Run</th>
                    <th>Registered</th>
                  </tr>
                </thead>
                <tbody>
                  {data.versions.map((version) => (
                    <tr
                      key={version.version}
                      className={version.is_production ? "is-production" : ""}
                    >
                      <td>v{version.version}</td>
                      <td>
                        {version.aliases.length > 0
                          ? version.aliases.map((alias) => (
                              <span className="pill" key={alias} style={{ marginRight: 4 }}>
                                {alias}
                              </span>
                            ))
                          : "—"}
                      </td>
                      <td>
                        {version.gates_passed !== null && version.gates_total !== null
                          ? `${version.gates_passed}/${version.gates_total}`
                          : "—"}
                      </td>
                      {TABLE_METRICS.map((key) => (
                        <td className="num" key={key}>
                          {metric(version.metrics[key], 4)}
                        </td>
                      ))}
                      <td className="faint">{shortId(version.mlflow_run_id, 8)}</td>
                      <td className="faint">{when(version.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Quality gates"
        note="Names come from ml_platform.promotion.gates, in the order the promotion code runs them. A blocking failure rejects the candidate and the incumbent stays exactly where it was."
      >
        <SectionBody state={gates} rows={4}>
          {(items) => (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Gate</th>
                    <th>Configured here</th>
                    <th>Why it blocks</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((gate) => (
                    <tr key={gate.name}>
                      <td>{gate.name}</td>
                      <td>
                        <span className={`pill ${gate.configured ? "healthy" : ""}`}>
                          {gate.configured ? "yes" : "not configured"}
                        </span>
                      </td>
                      <td className="muted" style={{ fontFamily: "var(--sans)" }}>
                        {gate.rationale}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </SectionBody>
      </Panel>

      <Caveat>
        <strong>A worse or unsafe model never replaces a working one automatically.</strong>{" "}
        Retraining, drift response and canary completion all converge on the same gate set; none
        of them has a path that writes the production alias without passing it.
      </Caveat>
    </>
  );
}
