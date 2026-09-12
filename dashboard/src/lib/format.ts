// Formatting only. No derivation, no inference: if the API did not send a
// number, these render an em dash rather than a zero, because zero is a
// measurement and "we don't know" is not.

export const UNKNOWN = "—";

export function metric(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? UNKNOWN
    : value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? UNKNOWN
    : `${(value * 100).toFixed(digits)}%`;
}

export function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return UNKNOWN;
  return value < 1 ? `${(value * 1000).toFixed(1)} ms` : `${value.toFixed(3)} s`;
}

export function rate(value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? UNKNOWN
    : `${value.toFixed(2)}/s`;
}

export function count(value: number | null | undefined): string {
  return value === null || value === undefined ? UNKNOWN : Math.round(value).toLocaleString();
}

export function when(iso: string | null | undefined): string {
  if (!iso) return UNKNOWN;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return UNKNOWN;
  return date.toISOString().replace("T", " ").slice(0, 19) + "Z";
}

export function shortId(value: string | null | undefined, length = 12): string {
  return value ? value.slice(0, length) : UNKNOWN;
}

/** Metric keys as the run records name them, mapped to display labels only. */
export const METRIC_LABELS: Record<string, string> = {
  average_precision: "Average precision",
  roc_auc: "ROC AUC",
  recall_at_capacity: "Recall @ 10%",
  precision_at_capacity: "Precision @ 10%",
  brier_score: "Brier score",
  brier_skill_score: "Brier skill",
};
