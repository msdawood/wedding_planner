"""Celery tasks: durable execution of the LangGraph workflow + out-of-band evaluation.

Crash recovery: the graph is compiled with a Postgres checkpointer using one checkpoint thread
per run. If the worker dies mid-run, Celery redelivers the task (acks_late +
reject_on_worker_lost) and `_execute` resumes from the last completed step instead of
starting over (and re-paying for the finished steps).
"""
import asyncio
import json
import logging

import redis
from celery.exceptions import SoftTimeLimitExceeded
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import observability
import run_store
import settings
from celery_app import EVAL_TASK, RUN_TASK, celery_app
from errors import BudgetExceeded, TransientError, is_budget_error, is_transient_error

log = logging.getLogger(__name__)

_redis: redis.Redis | None = None
_schema_ready = False
_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run a coroutine on ONE long-lived event loop per worker process.

    `asyncio.run()` creates and then CLOSES a loop on every call. LangChain's OpenAI client keeps a
    process-wide cache of HTTP connections, so the 2nd task in a worker process would reuse
    connections bound to the 1st task's closed loop and fail with "RuntimeError: Event loop is closed".
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    try:
        return _loop.run_until_complete(coro)
    finally:
        # After a time limit / crash, don't let half-finished tasks leak into the next job.
        pending = [t for t in asyncio.all_tasks(_loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            _loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


# --------------------------------------------------------------------------- the run
async def _execute(run_id: str, user_id: str, thread_id: str, prompt: str) -> dict:
    """Run (or resume) the graph; returns the final state values."""
    global _schema_ready
    from wedding_graph import build_graph          # imported lazily: heavy, and only workers need it

    r = get_redis()
    handler = observability.new_handler(run_id)
    config = {
        # One checkpoint thread per run => state can never leak between runs or users.
        "configurable": {"thread_id": f"{user_id}:{thread_id}:{run_id}"},
        "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
        "callbacks": [handler] if handler else [],
        "metadata": observability.trace_metadata(user_id, thread_id, run_id),
    }

    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as saver:
        if not _schema_ready:
            await saver.setup()                     # idempotent: creates the checkpoint tables
            _schema_ready = True
        graph = build_graph(checkpointer=saver)

        snapshot = await graph.aget_state(config)
        if snapshot.next:                            # unfinished checkpoint from a crashed attempt
            graph_input = None
            run_store.add_event(r, run_id, "resumed", {"next": list(snapshot.next)})
        else:
            graph_input = {"run_id": run_id, "request": prompt}

        async for update in graph.astream(graph_input, config, stream_mode="updates"):
            for node in update:
                run_store.add_event(r, run_id, "node_complete", {"node": node})

        values = (await graph.aget_state(config)).values

    observability.flush()
    return values


@celery_app.task(name=RUN_TASK, bind=True, max_retries=settings.TASK_MAX_RETRIES)
def execute_agent_workflow(self, run_id: str, user_id: str, thread_id: str, user_prompt: str):
    r = get_redis()
    attempt = self.request.retries
    run_store.set_status(r, run_id, run_store.RUNNING, attempt=attempt)
    run_store.add_event(r, run_id, "started", {"attempt": attempt})

    try:
        values = run_async(_execute(run_id, user_id, thread_id, user_prompt))
    except SoftTimeLimitExceeded:
        return _fail(r, run_id, "timeout")
    except Exception as exc:                          # noqa: BLE001
        if is_budget_error(exc):
            return _fail(r, run_id, "budget_exceeded")
        if is_transient_error(exc) and attempt < self.max_retries:
            delay = min(2 ** attempt * 5, 60)
            run_store.set_status(r, run_id, run_store.RETRYING, attempt=attempt + 1, error=type(exc).__name__)
            run_store.add_event(r, run_id, "retrying", {"attempt": attempt + 1, "in_seconds": delay})
            raise self.retry(exc=exc, countdown=delay)    # strict cap: max_retries
        log.exception("run %s failed", run_id)
        return _fail(r, run_id, f"{type(exc).__name__}: retries exhausted" if is_transient_error(exc)
                     else "internal_error")

    trace_id = observability.trace_id_for(run_id)
    outcome = values.get("status", "ok")
    status = {"ok": run_store.SUCCEEDED, "refused": run_store.REFUSED,
              "needs_info": run_store.NEEDS_INFO}.get(outcome, run_store.SUCCEEDED)
    result = {k: values.get(k) for k in
              ("plan", "flights", "venues", "playlist", "origin", "destination", "guest_count", "genre", "warnings")}

    run_store.set_status(r, run_id, status, result=result, trace_id=trace_id or "")
    run_store.add_event(r, run_id, "done", {"status": status})

    if status == run_store.SUCCEEDED and values.get("plan"):
        context = [values.get("flights", ""), values.get("venues", ""), values.get("playlist", "")]
        celery_app.send_task(EVAL_TASK, args=[run_id, user_prompt, values["plan"], context, trace_id])
    return status


def _fail(r, run_id: str, error: str) -> str:
    run_store.set_status(r, run_id, run_store.FAILED, error=error)
    run_store.add_event(r, run_id, "done", {"status": run_store.FAILED, "error": error})
    return run_store.FAILED


# --------------------------------------------------------------------------- evaluation
@celery_app.task(name=EVAL_TASK, bind=True, max_retries=1, soft_time_limit=300, time_limit=360)
def evaluate_agent_run(self, run_id: str, user_prompt: str, plan: str, context: list[str], trace_id: str | None):
    """Runs after the user already has their answer. Failures here never affect a run."""
    from evals import (HALLUCINATION_THRESHOLD, RELEVANCY_THRESHOLD, run_metrics)

    r = get_redis()
    try:
        res = run_metrics(user_prompt, plan, context)
    except Exception as exc:                          # noqa: BLE001
        if is_transient_error(exc) and self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        log.exception("evaluation of run %s failed", run_id)
        return None

    r.hset(run_store.run_key(run_id), "eval", json.dumps(res, default=str))

    # Continuous regression signal: rolling averages over the last 20 evaluated runs.
    averages = run_store.record_eval_scores(
        r, {"hallucination": res["hallucination_score"] or 0.0, "relevancy": res["relevancy_score"] or 0.0})
    # (the "hallucination" series is a groundedness score: 1.0 = no hallucination)
    observability.score(trace_id, "hallucination", res["hallucination_score"] or 0.0, res["hallucination_reason"])
    observability.score(trace_id, "relevancy", res["relevancy_score"] or 0.0, res["relevancy_reason"])

    # Both metrics: higher is better. A rolling average under its minimum = quality regression.
    regression = (averages["hallucination"] < HALLUCINATION_THRESHOLD
                  or averages["relevancy"] < RELEVANCY_THRESHOLD)
    if regression:
        log.warning("REGRESSION SIGNAL rolling averages %s (run %s)", averages, run_id)
        observability.score(trace_id, "regression_alert", 1.0, json.dumps(averages))
    observability.flush()
    log.info("eval run=%s %s rolling=%s", run_id, res, averages)
    return {**res, "rolling": averages, "regression": regression}
