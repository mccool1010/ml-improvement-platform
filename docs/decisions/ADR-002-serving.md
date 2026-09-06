# ADR-002: KServe is the canonical serving path

- **Status:** Accepted
- **Date:** 2026-09-06
- **Milestone:** M0, implemented at M11

## Context

The original repository sketch contained two ways to serve the same model: a plain
Kubernetes Deployment with a Service and Ingress in front of a FastAPI process,
and a KServe `InferenceService`. Both would have carried production traffic.

Two production serving paths for one model is a liability, not a demonstration of
breadth. They drift apart, they need separate rollout and rollback procedures,
and a canary proven on one says nothing about the other. It also invites the
failure this project is meant to argue against: adding a technology because it
looks impressive rather than because it does a job.

## Decision

**KServe is the canonical model-serving path in Kubernetes.** Traffic splitting,
canary rollout, revision management and rollback are KServe's responsibility.

**FastAPI stays, in a narrower role.** It is the application and API layer, and
the local development server. It owns request and response validation, the
health and readiness endpoints, model-version metadata, the business-facing
request shape, and the Prometheus metrics for the application tier. It calls the
KServe endpoint for predictions rather than loading the model itself.

The plain Kubernetes `Deployment` manifests remain only for the FastAPI
application tier. There will be no second Deployment that serves the model
directly.

## Why this way round

- **Canary and rollback are the project's requirement, and they are KServe's
  native capability.** Doing percentage traffic splits by hand with two
  Deployments and a Service selector is possible, but it reimplements badly what
  KServe does declaratively, and the rollback story becomes bespoke.
- **The model artifact stays decoupled from the application image.** A promoted
  model version changes an `InferenceService` revision, not the application
  container. That keeps the promotion path in ADR-003 independent of application
  releases.
- **An API layer still earns its place.** Raw KServe expects a tensor-shaped
  payload. The consumers of this system send a loan application. Validation,
  domain error messages, model-version reporting and request logging belong in a
  service that speaks the domain language, and FastAPI does that well.

## Consequences

- Local development runs FastAPI with a locally loaded model, and Kubernetes runs
  FastAPI in front of KServe. That difference is real and must be covered by an
  integration test against a running `InferenceService`, not assumed away.
- The application tier and the model tier scale and roll out independently, which
  is desirable but means two things to observe. Metrics must carry the served
  model version so a latency or error change can be attributed to the right tier.
- KServe requires Knative and a service mesh or equivalent networking layer. That
  is a real operational cost, accepted because canary traffic management is a
  stated project requirement rather than an optional extra.
- Should KServe prove unworkable in the local cluster, the fallback is a single
  Deployment serving the model behind the same FastAPI contract. That fallback
  must be recorded here as a status change, not adopted silently.
