// The one place the dashboard talks to the platform.
//
// Every call goes to the FastAPI aggregation layer at /platform. The browser
// never talks to MLflow, Prometheus or the Kubernetes API: that would need
// credentials and knowledge of their internals, and would make the dashboard a
// second client of systems that already have one.
//
// Nothing here supplies a default on failure. A section that cannot be loaded
// stays absent so the UI can say so, because a plausible placeholder is
// indistinguishable from real data to whoever is reading it.

export type Status = "healthy" | "degraded" | "unavailable";

export interface ComponentHealth {
  name: string;
  status: Status;
  detail: string | null;
  owns: string | null;
}

export interface PlatformHealth {
  status: Status;
  components: ComponentHealth[];
  can_serve: boolean;
}

export interface ProductionModel {
  available: boolean;
  detail: string | null;
  name: string | null;
  version: string | null;
  alias: string | null;
  feature_set: string | null;
  mlflow_run_id: string | null;
  platform_run_id: string | null;
  decision_split: string | null;
  gates_passed: number | null;
  gates_total: number | null;
  git_revision: string | null;
  dataset_sha256: string | null;
  metrics: Record<string, number>;
  served_by: string | null;
  decision_threshold: number | null;
  threshold_source: string | null;
}

export interface VersionSummary {
  version: string;
  aliases: string[];
  is_production: boolean;
  mlflow_run_id: string | null;
  gates_passed: number | null;
  gates_total: number | null;
  metrics: Record<string, number>;
  created_at: string | null;
}

export interface PromotionHistory {
  available: boolean;
  detail: string | null;
  registered_model: string | null;
  production_version: string | null;
  versions: VersionSummary[];
  unregistered_candidates: number | null;
}

export interface FeatureDrift {
  feature: string;
  psi: number;
  drifted: boolean;
}

export interface DriftState {
  available: boolean;
  detail: string | null;
  decision: string | null;
  drift_detected: boolean | null;
  n_drifted: number | null;
  n_features: number | null;
  max_psi: number | null;
  threshold_psi: number | null;
  min_drifted_features: number | null;
  window_start: string | null;
  window_end: string | null;
  scenario: string | null;
  checked_at: string | null;
  top_features: FeatureDrift[];
  retraining: Record<string, unknown>;
  performance_signal_available: boolean;
  performance_note: string;
}

export interface CanaryState {
  available: boolean;
  detail: string | null;
  traffic_percent: number;
  active: boolean;
  incumbent_version: string | null;
  candidate_version: string | null;
  last_decision: string | null;
  last_reason: string | null;
  decided_at: string | null;
  signals: Record<string, number>;
  rollback_signals: string[];
  accuracy_used_as_signal: boolean;
  signal_note: string;
}

export interface FailureScenario {
  scenario: string;
  mode: string;
  invariants: string[];
  passed: boolean | null;
  observed: string | null;
}

export interface FailureInvariant {
  name: string;
  falsified_by: string;
}

export interface FailureState {
  available: boolean;
  detail: string | null;
  scenarios: FailureScenario[];
  invariants: FailureInvariant[];
  evidence_available: boolean;
  last_run_id: string | null;
  n_passed: number | null;
}

export interface Observability {
  available: boolean;
  detail: string | null;
  request_rate: number | null;
  error_ratio: number | null;
  latency_p50_seconds: number | null;
  latency_p95_seconds: number | null;
  model_tier_p95_seconds: number | null;
  applications_scored: number | null;
  grafana_url: string | null;
  jaeger_url: string | null;
  prometheus_url: string | null;
}

export interface QualityGate {
  name: string;
  rationale: string;
  configured: boolean;
}

export interface PlatformEvent {
  at: string;
  kind: string;
  summary: string;
  outcome: string | null;
}

export interface LifecycleStage {
  stage: string;
  component: string;
  milestone: string;
  state: "observed" | "implemented" | null;
}

export class ApiError extends Error {}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) {
    throw new ApiError(`${path} returned ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => get<PlatformHealth>("/platform/health"),
  model: () => get<ProductionModel>("/platform/model"),
  lifecycle: () => get<LifecycleStage[]>("/platform/lifecycle"),
  promotions: () => get<PromotionHistory>("/platform/promotions"),
  drift: () => get<DriftState>("/platform/drift"),
  canary: () => get<CanaryState>("/platform/canary"),
  failure: () => get<FailureState>("/platform/failure"),
  observability: () => get<Observability>("/platform/observability"),
  gates: () => get<QualityGate[]>("/platform/gates"),
  events: () => get<PlatformEvent[]>("/platform/events"),
};
