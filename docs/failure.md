# Failure engineering

Breaking real components on purpose, checking that the damage stopped where it
should have, and writing down what was observed.

```bash
kubectl -n ml-platform port-forward svc/inference-api 8080:80 &
kubectl -n ml-platform port-forward svc/mlflow 5000:5000 &

python -m ml_platform failure --base-url http://localhost:8080 \
    --tracking-uri http://localhost:5000            # all six
python -m ml_platform failure --offline             # only what breaks nothing
python -m ml_platform failure --scenario canary_failure
```

Not a chaos framework. Six named scenarios, `kubectl scale` as the injector, and
a report. Adding Argo, Litmus or a mesh to prove six things would be exactly the
technology-for-its-own-sake this project argues against.

**Every injector has a matching restore in a `finally`.** A harness that can
leave the cluster broken is a worse liability than the failures it tests for.

## Invariants

An invariant here is narrow enough to be falsified by a specific observation.
"The platform is resilient" is not one; "a request never receives a fabricated
prediction" is.

The idea running through all of them: **failing closed is success.** A service
answering 503 because its model tier is gone has behaved correctly. One
answering 200 with a number it invented has not, and that is far harder to
notice — a loan decision made on a fabricated score is indistinguishable from a
correct one until much later.

| Invariant | Falsified by |
|---|---|
| `no_fabricated_predictions` | any 200-with-a-score while the model tier is down |
| `production_alias_unchanged` | the alias moving during a failure |
| `failed_candidate_not_production` | a rejected candidate being registered or aliased |
| `canary_rollback_restores_incumbent` | any request still reaching the canary after rollback |
| `telemetry_failure_isolated` | an inference failing because a metrics backend is down |
| `inference_unaffected` | the same, for a non-telemetry dependency |
| `recovery_returns_known_state` | coming back unhealthy, **or** coming back serving a different model |
| `dependency_failure_is_explicit` | an operation succeeding against a dependency that is not there |

Each is tested in both directions in `tests/integration/test_failure.py` — an
invariant that cannot fail proves nothing.

## Failure matrix

| Scenario | Injected | Blast radius | Unaffected | Invariants |
|---|---|---|---|---|
| `model_serving_failure` | predictor → 0 replicas | predictions unavailable; readiness false so the Service drops the pods | liveness, the process, the registry | no fabricated predictions; explicit failure; alias unchanged |
| `mlflow_dependency_failure` | mlflow → 0 replicas | registry reads/writes gone: promotion, canary completion, drift logging; **a new pod would start unready** | inference, the alias | inference unaffected; explicit failure; recovery |
| `bad_candidate_promotion` | a crippled candidate through `run_promotion` | **none** | everything | failed candidate not production; alias unchanged |
| `canary_failure` | 40% traffic to a canary, then canary → 0 replicas | ~40% of requests fail, by design | the incumbent's share, the alias | no fabricated predictions; rollback restores incumbent; alias unchanged |
| `telemetry_failure` | prometheus and jaeger → 0 replicas | no metrics, no traces; canary evaluation could not read signals | inference, readiness, the registry | telemetry isolated; alias unchanged |
| `restart_recovery` | rollout restart of the predictor and the API | requests refused during the rollout; in-process counters reset | the alias, the routing allocation | recovery; routing unchanged; identical input gives identical score |

Five break real components. **One — `bad_candidate_promotion` — is a controlled
double**, and its report says `controlled-double` so nobody has to guess. It runs
the genuine promotion pipeline, the genuine seven gates and the genuine
registration code against a synthetic register and an isolated store, because
destroying the live registry to prove a bad model is rejected would produce worse
evidence, not better.

## Live evidence

```
model_serving_failure
  /predict [503] with no scores; /ready 503; /health 200
  detail: "the model tier could not be reached: could not reach the predictor at
           http://sba-loan-default-predictor: [Errno 111] Connection refused"
  recovery: scaled back to 1; a prediction returned 0.003955
  3/3 invariants held

mlflow_dependency_failure
  /predict [200] with 8 scores; a registry lookup returned None
  recovery: mlflow ready; model identity unchanged
  3/3 invariants held

canary_failure
  with the canary dead, 7 of 20 requests failed (40% allocation)
  after rollback, 0 of 20 reached the canary
  production v1 before and after
  3/3 invariants held

telemetry_failure
  12 of 12 requests served normally; /metrics 200; /ready 200
  2/2 invariants held

restart_recovery
  after restart /predict 200 with 0.003955 (was 0.003955)
  canary allocation 0.0% -> 0.0%
  4/4 invariants held

bad_candidate_promotion  [controlled-double]
  [REJECT] 1/7 gates passed, 6 blocking failures:
    min_improvement, minimum_metric, roc_auc_regression,
    calibration, recall_regression, reproducibility
  production v1 -> v1, nothing registered
  2/2 invariants held
```

## Two defects the harness found in itself

Both were in the exercise, not the platform, and both would have produced
confident nonsense.

**The registry check was asking the wrong registry.** `_production_version`
resolves through `config.tracking_uri`, which on a laptop is the *local* SQLite
store. The first run of the MLflow scenario scaled the cluster's MLflow to zero
and then asked the laptop whether the registry was alive; it said yes, and
`dependency_failure_is_explicit` correctly reported a violation. Live runs now
set `MLFLOW_TRACKING_URI` at the cluster's server. The invariant caught this,
which is the entire argument for making invariants strict.

**Probing over a port-forward could not survive a restart.** Changing the canary
allocation restarts the API, and the forward dies with the pod it was attached
to. The first canary run recorded "40 of 40 requests failed" — every one a
connection error indistinguishable from the failure being studied, and a number
that looked like evidence. Scenarios that restart anything now probe from inside
a pod through the real Service; the corrected run shows 7 of 20.

No defect was found in M6–M14 itself, and nothing in those milestones was changed
to make a scenario pass.

## Known behaviour, observed and not fixed

**A pod that starts while MLflow is down stays unready.** The API resolves its
model once at startup and does not retry, so if the registry is unavailable at
that moment the pod never becomes ready, even after MLflow returns; it needs a
restart. This is recorded in the `mlflow_dependency_failure` blast radius.

It is deliberately **not** fixed here. It fails *closed* — the pod withholds
traffic rather than serving something wrong — so it is an availability
limitation, not a safety one, and load-once is the M7 decision that keeps a
registry lookup off the request path. Changing it to make a scenario read better
would be modifying the system to suit the test.

**A registry read against a dead server is slow to fail** from the CLI: the
MLflow client retries with backoff, and the bounds that fix this
(`MLFLOW_HTTP_REQUEST_MAX_RETRIES`) are set on the pods, not on a laptop. The
MLflow scenario therefore takes minutes rather than seconds.

## Limitations

- **Six scenarios, not an exhaustive fault model.** Nothing here tests partial
  network partitions, disk exhaustion, slow-but-alive dependencies, or
  corrupted artifacts. Scaling to zero is a clean failure; the messy ones are
  harder to inject and were not attempted.
- **Nothing runs these automatically.** They are a command, not a schedule.
- **The blast radius is asserted by observation, not proven.** "The registry is
  unaffected" means it was checked and had not changed, not that nothing could
  have changed it.
- **Recovery is measured immediately.** A scenario that damaged something which
  only surfaces hours later would be recorded as recovered.
