import type { ReactNode } from "react";

import type { Status } from "../lib/api";
import { METRIC_LABELS, metric as fmtMetric } from "../lib/format";

export function Panel(props: {
  title: string;
  status?: Status | null;
  aside?: ReactNode;
  note?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <header>
        {props.status ? <span className={`dot ${props.status}`} aria-hidden /> : null}
        <h3>{props.title}</h3>
        <span className="spacer" />
        {props.aside}
      </header>
      <div className="panel-body">{props.children}</div>
      {props.note ? <div className="panel-note">{props.note}</div> : null}
    </section>
  );
}

export function StatusPill({ status, label }: { status: Status; label?: string }) {
  return (
    <span className={`pill ${status}`}>
      <span className={`dot ${status}`} aria-hidden />
      {label ?? status}
    </span>
  );
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div role="status" aria-label="loading">
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton" key={i} />
      ))}
    </div>
  );
}

/**
 * What a panel shows when there is nothing to show.
 *
 * Deliberately not an error state: a drift check that has never been run and an
 * MLflow server that is down are both "no data", and in both cases the honest
 * thing is to name the reason and, where one exists, the command that would
 * produce the data.
 */
export function Unavailable({
  reason,
  hint,
  command,
}: {
  reason: string | null | undefined;
  hint?: string;
  command?: string;
}) {
  return (
    <div className="unavailable-block">
      <strong>Not available.</strong> {reason ?? "no detail was reported"}
      {hint ? <div style={{ marginTop: 8 }}>{hint}</div> : null}
      {command ? (
        <div style={{ marginTop: 8 }}>
          <code>{command}</code>
        </div>
      ) : null}
    </div>
  );
}

/** Renders a section: loading, then a fetch failure, then the content. */
export function SectionBody<T>(props: {
  state: { data: T | null; error: string | null; loading: boolean };
  /** Read a per-section availability flag off the payload, if it has one. */
  unavailable?: (data: T) => { detail: string | null } | null;
  command?: string;
  rows?: number;
  children: (data: T) => ReactNode;
}) {
  const { data, error, loading } = props.state;
  if (loading && data === null) return <Loading rows={props.rows} />;
  if (error !== null) {
    return <Unavailable reason={`the platform API could not be reached (${error})`} />;
  }
  if (data === null) return <Unavailable reason="the platform API returned nothing" />;
  const missing = props.unavailable?.(data);
  if (missing) return <Unavailable reason={missing.detail} command={props.command} />;
  return <>{props.children(data)}</>;
}

export function KeyValue({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([key, value]) => (
        <div key={key} style={{ display: "contents" }}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function MetricGrid({
  metrics,
  digits = 4,
}: {
  metrics: Record<string, number>;
  digits?: number;
}) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return <p className="muted">No metrics were recorded against this version.</p>;
  }
  const order = Object.keys(METRIC_LABELS);
  entries.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  return (
    <div className="metrics">
      {entries.map(([key, value]) => (
        <div className="metric" key={key}>
          <div className="label">{METRIC_LABELS[key] ?? key}</div>
          <div className="value">{fmtMetric(value, digits)}</div>
          <div className="sub">{key}</div>
        </div>
      ))}
    </div>
  );
}

/** The project's non-negotiable distinctions, stated where they apply. */
export function Caveat({ children }: { children: ReactNode }) {
  return <div className="caveat">{children}</div>;
}
