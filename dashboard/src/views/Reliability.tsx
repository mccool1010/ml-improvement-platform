import { api } from "../lib/api";
import { Caveat, Panel, SectionBody } from "../components/primitives";
import { useSection } from "../lib/useSection";

export function Reliability() {
  const failure = useSection(api.failure);
  const health = useSection(api.health);

  return (
    <>
      <div className="view-head">
        <h2>Reliability &amp; failure engineering</h2>
        <p>
          Real components broken on purpose, with the damage checked against named invariants.
          The governing idea is that <strong>failing closed is success</strong>: a service
          answering 503 because its model tier is gone has behaved correctly, and one answering
          200 with a number it invented has not.
        </p>
      </div>

      <Panel
        title="Scenario catalogue"
        status={health.data?.status ?? null}
        note={failure.data?.detail ?? undefined}
      >
        <SectionBody
          state={failure}
          rows={4}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Mode</th>
                    <th>Command</th>
                  </tr>
                </thead>
                <tbody>
                  {data.scenarios.map((scenario) => (
                    <tr key={scenario.scenario}>
                      <td>{scenario.scenario}</td>
                      <td>
                        <span
                          className={`pill ${
                            scenario.mode === "live-kubernetes" ? "healthy" : "degraded"
                          }`}
                        >
                          {scenario.mode}
                        </span>
                      </td>
                      <td className="faint">
                        python -m ml_platform failure --scenario {scenario.scenario}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </SectionBody>
      </Panel>

      <Panel
        title="Invariants"
        note="Names come from ml_platform.failure.invariants. Each is tested in both directions: an invariant that cannot fail proves nothing."
      >
        <SectionBody
          state={failure}
          rows={4}
          unavailable={(data) => (data.available ? null : { detail: data.detail })}
        >
          {(data) => (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Invariant</th>
                    <th>Falsified by</th>
                  </tr>
                </thead>
                <tbody>
                  {data.invariants.map((invariant) => (
                    <tr key={invariant.name}>
                      <td>{invariant.name}</td>
                      <td className="muted" style={{ fontFamily: "var(--sans)" }}>
                        {invariant.falsified_by}
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
        <strong>Five scenarios break real components; one is a controlled double</strong> and
        says so in its own report. Destroying the live registry to prove that a bad model is
        rejected would produce worse evidence, not better, so that one runs the genuine promotion
        pipeline and the genuine gates against a synthetic register in an isolated store.
        Recorded results, including the two defects the harness found in itself, are in{" "}
        <span className="mono">docs/failure.md</span>.
      </Caveat>
    </>
  );
}
