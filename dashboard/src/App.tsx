import { useEffect, useState } from "react";

import { api } from "./lib/api";
import { useSection } from "./lib/useSection";
import { Canary } from "./views/Canary";
import { Drift } from "./views/Drift";
import { Models } from "./views/Models";
import { Observability } from "./views/Observability";
import { Overview } from "./views/Overview";
import { Reliability } from "./views/Reliability";

const VIEWS = [
  { id: "overview", label: "Overview", render: () => <Overview /> },
  { id: "models", label: "Models & promotion", render: () => <Models /> },
  { id: "drift", label: "Drift & retraining", render: () => <Drift /> },
  { id: "canary", label: "Canary & rollback", render: () => <Canary /> },
  { id: "reliability", label: "Reliability", render: () => <Reliability /> },
  { id: "observability", label: "Observability", render: () => <Observability /> },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

function viewFromHash(): ViewId {
  const id = window.location.hash.replace(/^#\/?/, "");
  return VIEWS.some((view) => view.id === id) ? (id as ViewId) : "overview";
}

export function App() {
  const [active, setActive] = useState<ViewId>(viewFromHash);
  const health = useSection(api.health);

  // The hash is the only routing this needs. A router library for six static
  // views would be a dependency carrying no weight.
  useEffect(() => {
    const onHashChange = () => setActive(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const current = VIEWS.find((view) => view.id === active) ?? VIEWS[0];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>ML Platform</h1>
          <p>SBA loan default · production</p>
        </div>
        <nav className="nav">
          {VIEWS.map((view) => (
            <button
              key={view.id}
              type="button"
              aria-current={view.id === active ? "page" : undefined}
              onClick={() => {
                window.location.hash = `#/${view.id}`;
                setActive(view.id);
              }}
            >
              {view.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          {health.data ? (
            <>
              <span className={`dot ${health.data.status}`} aria-hidden /> {health.data.status}
              <div style={{ marginTop: 4 }}>
                {health.data.can_serve ? "serving" : "not serving"}
              </div>
            </>
          ) : health.error ? (
            <>
              <span className="dot unavailable" aria-hidden /> API unreachable
            </>
          ) : (
            "checking…"
          )}
        </div>
      </aside>
      <main className="main">{current.render()}</main>
    </div>
  );
}
