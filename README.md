# Wedding Planner: production stack

```
client ──JWT──▶ FastAPI gateway ──enqueue──▶ Redis ──▶ Celery worker ──▶ LangGraph (Postgres checkpoints)
   ▲   202 + run_id (ms)   │ status & events ◀─────────────┘                 │ every LLM call
   └───── SSE / GET ───────┘                                                 ▼
                                   DeepEval (queue `evals`)           LiteLLM proxy ──▶ Anthropic
                                   Langfuse v3 (OTel traces + eval scores) ◀── worker spans
```

| # | Requirement | Where |
|---|---|---|
| 1 | FastAPI gateway: JWT, sanitised payloads, instant tracker | `src/api.py`, `src/auth.py` |
| 2 | Redis + Celery: offload, memory bounds, strict retries | `src/celery_app.py`, `src/tasks.py` |
| 3 | DB-independent LangGraph, compaction, safety routing | `src/wedding_graph.py`, `src/compaction.py` |
| 4 | Postgres checkpointer, crash recovery | `src/tasks.py` (`AsyncPostgresSaver`) |
| 5 | LiteLLM budget gateway | `litellm-config.yaml`, `litellm_init` in compose, `src/llm.py` |
| 6 | DeepEval post-run metrics, regression signal | `src/evals.py`, `evaluate_agent_run` |
| 7 | Langfuse tracing (OpenTelemetry) | `src/observability.py` |

## Run it

```bash
docker compose up --build -d
docker compose ps                      # everything should become healthy / "exited 0" for litellm_init
```
Set `TAVILY_API_KEY` in `.env` first (venue search). Langfuse UI: http://localhost:3000 (login = `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`).

```bash
# 1. get a token (API_USERNAME / API_PASSWORD from .env)
TOKEN=$(curl -s -X POST localhost:8000/auth/token -d "username=admin&password=$API_PASSWORD" | jq -r .access_token)

# 2. submit: returns 202 immediately with a tracker
curl -s -X POST localhost:8000/runs -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_message":"Wedding in Paris from London for 100 guests, jazz playlist"}'
# {"run_id":"…","thread_id":"…","status":"queued","status_url":"/runs/…","stream_url":"/runs/…/stream"}

# 3a. poll
curl -s localhost:8000/runs/$RUN_ID -H "Authorization: Bearer $TOKEN"
# 3b. or stream progress (replays from the start; resume with the Last-Event-ID header)
curl -N localhost:8000/runs/$RUN_ID/stream -H "Authorization: Bearer $TOKEN"
```
Statuses: `queued → running → (retrying) → succeeded | failed | refused | needs_info`. `eval` appears on the run a little after it succeeds.

## Behaviour worth knowing

* **Gateway never runs the agent.** Payload is Pydantic-validated (unknown fields rejected, `thread_id` charset, length cap), unicode-normalised and stripped of control/zero-width characters. 10 runs/min/user by default. A user can only read their own runs (others get 404).
* **Retries.** Transient failures (provider 5xx, timeouts, proxy rate limits) retry with backoff, at most `TASK_MAX_RETRIES` (3) times. Budget exhaustion and unexpected errors never retry. Hard limits: 600 s soft / 660 s hard per run.
* **Memory.** Worker child processes are recycled after 20 tasks or 1 GB; container is capped at 2 GB.
* **Crash recovery.** One checkpoint thread per run. If a worker process dies, Celery requeues the job (`acks_late` + `reject_on_worker_lost`) and the run **resumes from the last checkpoint**; steps that already finished are not re-run or re-billed. *If the whole container dies*, Redis only redelivers after the visibility timeout (`TASK_TIME_LIMIT + 300 s`, about 16 min).
* **Spend control.** The worker holds only a LiteLLM virtual key (`LITELLM_WORKER_KEY`) with `LITELLM_MAX_BUDGET` per `LITELLM_BUDGET_DURATION` plus TPM/RPM limits; the Anthropic key exists only inside the proxy container. `litellm-config.yaml` adds a global ceiling (`max_budget: 25`). When the budget is hit, runs fail with `budget_exceeded` (no retries).
* **Safety routing in the graph.** Prefilter + LLM classifier → `refuse` path; missing/absurd details → `needs_info` before any paid tool call; a failing specialist degrades to an "unavailable" note; prompt-injection hardening in the synthesis prompt; the playlist SQL tool is read-only.
* **Compaction.** Tool outputs are capped (4 000 chars) and older ones shrunk (500) before each model call, so a 12-search venue loop can't blow the context window.
* **Evals.** DeepEval judges (same model, through the gateway) run on a separate queue. Scores go to Redis, Langfuse (attached to the run's trace) and a rolling 20-run average; a drop below threshold logs `REGRESSION SIGNAL` and adds a `regression_alert` score in Langfuse. DeepEval ≥ 4 scores *higher = better* for hallucination too (1.0 = fully grounded).
* **Langfuse versions.** Server v3 (this stack) requires Python SDK **v4** (pinned `langfuse<5`). SDK v5 needs Langfuse server ≥ 4.59, so upgrade both together.

## Known limitations (v0.1.0)

This is a working reference stack, not a hardened product. Know these before relying on it:

* **Authentication is tutorial-level.** One user from `API_USERNAME` / `API_PASSWORD`, HS256 JWTs signed with a shared secret. No user store, roles, refresh tokens or key rotation. For real use, put an identity provider (Cognito, Auth0, Keycloak) in front and validate its tokens.
* **The eval judge is the same model that wrote the answer.** Scores measure whether the plan is grounded in the specialist reports, not whether those reports are true, and a model can be lenient with its own output. Use a separate, stronger judge model and a small set of known-bad test cases before trusting the regression signal.
* **Checkpoints are never cleaned up.** Each run keeps its LangGraph checkpoint rows in Postgres forever. Add a retention job before running this at volume.
* **Mixed currencies in the plan.** Playlist costs come from the Chinook database in dollars, venue prices come from web search in whatever currency the source uses, and flights from Kiwi. The plan does not convert them.
* **Whole-container loss is slow to recover.** Redis redelivers an unfinished task only after the visibility timeout (about 16 minutes).
* **Spend limits are only as good as the price table.** Budgets are computed from LiteLLM's model price list and token counts; check provider billing for the real figure.
* **Local-only defaults.** Services bind to `127.0.0.1`, there is no TLS, and the compose file is not a deployment recipe.
* **Several tuning values are hard-coded.** Injection-filter regexes and prompts live in the source and are not yet configurable.

## Acknowledgements

Based on the wedding planner multi-agent example from LangChain Academy
([langchain-ai/lca-lc-foundations](https://github.com/langchain-ai/lca-lc-foundations), MIT licensed),
converted to LangGraph and extended with a FastAPI gateway, Celery workers, a LiteLLM spend gateway,
DeepEval metrics and Langfuse tracing. The sample music data is the
[Chinook database](https://github.com/lerocha/chinook-database). See `THIRD_PARTY_NOTICES.md`.

## Tests

```bash
pip install -r requirements.txt pytest pytest-asyncio fakeredis httpx
python -m pytest tests -q          # unit + API tests; integration tests need Postgres/Redis (see test_tasks_integration.py)
```
`tests/e2e_worker.py` + `tests/e2e_smoke.py` run the whole pipeline (real Celery worker, API, Redis, Postgres) with a fake LLM.

## Local development without Docker
`python src/wedding_graph.py` (run from the project folder) still works with just `ANTHROPIC_API_KEY`; with `LITELLM_BASE_URL` unset it calls Anthropic directly.