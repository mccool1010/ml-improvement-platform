import { vi } from "vitest";

import type {
  CanaryState,
  DriftState,
  FailureState,
  LifecycleStage,
  Observability,
  PlatformEvent,
  PlatformHealth,
  ProductionModel,
  PromotionHistory,
  QualityGate,
} from "./lib/api";

/**
 * Fixtures shaped like the real payloads, with the real field names.
 *
 * Values are placeholders — the tests never assert that a number is correct,
 * only that the UI renders what it was given and says nothing when it was
 * given nothing.
 */

export const health: PlatformHealth = {
  status: "degraded",
  can_serve: true,
  components: [
    {
      name: "API",
      status: "healthy",
      detail: "this process is serving",
      owns: "request validation, thresholds, version reporting",
    },
    {
      name: "Model serving",
      status: "healthy",
      detail: "serving via KServe",
      owns: "model artifact, feature building, scoring",
    },
    {
      name: "MLflow",
      status: "healthy",
      detail: "registry and run lineage reachable",
      owns: "experiment lineage, model registry, promotion history",
    },
    {
      name: "Prometheus",
      status: "unavailable",
      detail: "not reachable from the API",
      owns: "request metrics, canary signals",
    },
  ],
};

export const model: ProductionModel = {
  available: true,
  detail: null,
  name: "sba-loan-default",
  version: "1",
  alias: "production",
  feature_set: "v1",
  mlflow_run_id: "ab".repeat(16),
  platform_run_id: "20260101T000000Z-abcdef",
  decision_split: "validation",
  gates_passed: 7,
  gates_total: 7,
  git_revision: "0123456789abcdef",
  dataset_sha256: "cd".repeat(32),
  metrics: { average_precision: 0.1234, roc_auc: 0.7654 },
  served_by: "kserve:sba-loan-default",
  decision_threshold: 0.0456,
  threshold_source: "validation sweep",
};

export const promotions: PromotionHistory = {
  available: true,
  detail: null,
  registered_model: "sba-loan-default",
  production_version: "1",
  unregistered_candidates: 9,
  versions: [
    {
      version: "1",
      aliases: ["production"],
      is_production: true,
      mlflow_run_id: "ab".repeat(16),
      gates_passed: 7,
      gates_total: 7,
      metrics: { average_precision: 0.1234, roc_auc: 0.7654 },
      created_at: "2026-01-01T00:00:00+00:00",
    },
  ],
};

export const gates: QualityGate[] = [
  { name: "min_improvement", rationale: "must beat the incumbent by a stated margin", configured: true },
  { name: "latency", rationale: "a model too slow to serve is not an improvement", configured: true },
  { name: "reproducibility", rationale: "must be reproducible from its provenance", configured: false },
];

export const drift: DriftState = {
  available: true,
  detail: null,
  decision: "retrain",
  drift_detected: true,
  n_drifted: 3,
  n_features: 11,
  max_psi: 0.4123,
  threshold_psi: 0.1,
  min_drifted_features: 2,
  window_start: "2009-01-01",
  window_end: "2009-06-25",
  scenario: "post_crisis",
  checked_at: "2026-02-02T12:00:00+00:00",
  top_features: [
    { feature: "GrAppv", psi: 0.4123, drifted: true },
    { feature: "NoEmp", psi: 0.0421, drifted: false },
  ],
  retraining: { ran: true, candidate_run_id: "ff".repeat(16), promoted: false, window_rows: "12345" },
  performance_signal_available: false,
  performance_note: "Realised performance needs matured labels, which take 60 months.",
};

export const canary: CanaryState = {
  available: true,
  detail: null,
  traffic_percent: 0,
  active: false,
  incumbent_version: "1",
  candidate_version: null,
  last_decision: "rollback",
  last_reason: "candidate error rate exceeded the incumbent",
  decided_at: "2026-03-03T09:00:00+00:00",
  signals: { canary_error_ratio: 0.42, production_error_ratio: 0.001 },
  rollback_signals: ["candidate health", "HTTP error rate (absolute and vs incumbent)"],
  accuracy_used_as_signal: false,
  signal_note: "Rollback uses operational signals only.",
};

export const failure: FailureState = {
  available: true,
  detail: "Scenario evidence is written where the harness runs.",
  evidence_available: false,
  last_run_id: null,
  n_passed: null,
  scenarios: [
    { scenario: "canary_failure", mode: "live-kubernetes", invariants: [], passed: null, observed: null },
    {
      scenario: "bad_candidate_promotion",
      mode: "controlled-double",
      invariants: [],
      passed: null,
      observed: null,
    },
  ],
  invariants: [
    { name: "no_fabricated_predictions", falsified_by: "any 200-with-a-score while the model tier is down" },
    { name: "production_alias_unchanged", falsified_by: "the alias moving during a failure" },
  ],
};

export const observability: Observability = {
  available: true,
  detail: null,
  request_rate: 1.234,
  error_ratio: 0.01,
  latency_p50_seconds: 0.012,
  latency_p95_seconds: 0.098,
  model_tier_p95_seconds: 0.045,
  applications_scored: 4321,
  grafana_url: "http://localhost:3000",
  jaeger_url: "http://localhost:16686",
  prometheus_url: "http://localhost:9090",
};

export const lifecycle: LifecycleStage[] = [
  { stage: "train", component: "ml_platform.pipelines.train_pipeline", milestone: "M1-M2", state: "observed" },
  { stage: "canary", component: "ml_platform.serving.canary", milestone: "M14", state: "implemented" },
];

export const events: PlatformEvent[] = [
  {
    at: "2026-03-03T09:00:00+00:00",
    kind: "canary",
    summary: "candidate error rate exceeded the incumbent",
    outcome: "rollback",
  },
];

/** Every endpoint, keyed by path, as the happy path. */
export function allSections(): Record<string, unknown> {
  return {
    "/platform/health": health,
    "/platform/model": model,
    "/platform/promotions": promotions,
    "/platform/gates": gates,
    "/platform/drift": drift,
    "/platform/canary": canary,
    "/platform/failure": failure,
    "/platform/observability": observability,
    "/platform/lifecycle": lifecycle,
    "/platform/events": events,
  };
}

/**
 * Installs a `fetch` that answers from a path map.
 *
 * `overrides` may map a path to a thrown error (network failure) or to a
 * replacement payload, so a test can make exactly one dependency unavailable
 * and leave the rest healthy — which is how the platform actually fails.
 */
export function stubFetch(overrides: Record<string, unknown | Error> = {}) {
  const bodies = { ...allSections(), ...overrides };
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    const body = bodies[path];
    if (body instanceof Error) throw body;
    if (body === undefined) {
      return new Response("not found", { status: 404 });
    }
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
