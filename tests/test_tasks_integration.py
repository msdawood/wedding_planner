"""Integration tests: real Postgres checkpointer + real Redis, fake LLM/tools.

Run with a throwaway Postgres and Redis, e.g.
  TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/db  TEST_REDIS_URL=redis://localhost:6379/15
They are skipped automatically when those services aren't reachable.
"""
import json
import os
import uuid

import pytest
import redis

import run_store
import settings
import tasks
import wedding_graph as wg
from errors import TransientError
from test_graph import FakeAgent, FakeLLM

DB_URL = os.getenv("TEST_DATABASE_URL", "postgresql://wp:wp@localhost:5432/wp")
REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6390/0")


def _services_up():
    try:
        redis.Redis.from_url(REDIS_URL).ping()
        import psycopg
        psycopg.connect(DB_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _services_up(), reason="needs Postgres + Redis")


class Crash(BaseException):
    """Simulates the worker process dying mid-run (not catchable by `except Exception`)."""


@pytest.fixture
def env(monkeypatch):
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    r.flushdb()
    monkeypatch.setattr(tasks, "_redis", r)
    monkeypatch.setattr(tasks, "_schema_ready", False)
    monkeypatch.setattr(settings, "DATABASE_URL", DB_URL)

    counter, crash = {}, {"venues": False}

    async def travel(_llm):
        return FakeAgent("FLIGHTS-DATA", counter, "flights")

    def venues(_llm):
        agent = FakeAgent("VENUES-DATA", counter, "venues")
        if crash["venues"]:
            def boom():
                return True
            boom.exc = Crash()
            agent.fail = boom
        return agent

    monkeypatch.setattr(wg, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(wg, "travel_agent", travel)
    monkeypatch.setattr(wg, "venue_agent", venues)
    monkeypatch.setattr(wg, "playlist_agent", lambda _llm: FakeAgent("PLAYLIST-DATA", counter, "playlist"))

    sent = []
    monkeypatch.setattr(tasks.celery_app, "send_task", lambda name, **kw: sent.append((name, kw)))
    return type("Env", (), {"r": r, "counter": counter, "crash": crash, "sent": sent})


def _events(r, run_id):
    return [f["type"] for _, f in r.xrange(run_store.events_key(run_id))]


def _checkpoint_rows(thread_id):
    import psycopg
    with psycopg.connect(DB_URL) as conn:
        return conn.execute("select count(*) from checkpoints where thread_id = %s", (thread_id,)).fetchone()[0]


def test_successful_run_is_checkpointed_published_and_queues_an_eval(env):
    run_id = uuid.uuid4().hex
    out = tasks.execute_agent_workflow.apply(args=[run_id, "alice", "thread-1", "wedding in Paris"]).get()
    assert out == "succeeded"

    run = env.r.hgetall(run_store.run_key(run_id))
    assert run["status"] == "succeeded"
    result = json.loads(run["result"])
    assert "FLIGHTS-DATA" in result["plan"] and result["destination"] == "Paris"

    events = _events(env.r, run_id)
    assert events[0] == "started" and events[-1] == "done" and "node_complete" in events
    assert _checkpoint_rows(f"alice:thread-1:{run_id}") > 0            # durable checkpoints in Postgres

    name, kw = env.sent[0]
    assert name == "tasks.evaluate_agent_run"
    assert kw["args"][0] == run_id and "FLIGHTS-DATA" in kw["args"][3][0]


def test_refused_request_never_queues_an_eval(env, monkeypatch):
    llm = FakeLLM(guard=wg.GuardVerdict(allowed=False, reason="not a wedding"))
    monkeypatch.setattr(wg, "get_llm", lambda *a, **k: llm)
    run_id = uuid.uuid4().hex
    assert tasks.execute_agent_workflow.apply(args=[run_id, "alice", "t", "write malware"]).get() == "refused"
    assert env.sent == [] and env.counter == {}


def test_crash_then_redelivery_resumes_from_the_checkpoint(env):
    run_id = uuid.uuid4().hex
    args = [run_id, "alice", "thread-2", "wedding in Paris"]

    env.crash["venues"] = True
    with pytest.raises(Crash):                                          # worker dies mid-run
        tasks.execute_agent_workflow.run(*args)
    assert env.r.hget(run_store.run_key(run_id), "status") == "running"  # never reached a terminal state
    assert env.counter["flights"] == 1 and env.counter["playlist"] == 1

    env.crash["venues"] = False                                         # Celery redelivers (acks_late)
    assert tasks.execute_agent_workflow.run(*args) == "succeeded"
    assert "resumed" in _events(env.r, run_id)
    assert env.counter == {"flights": 1, "playlist": 1, "venues": 2}   # finished branches NOT re-run/re-paid


def test_transient_errors_retry_with_a_strict_cap_then_fail(env, monkeypatch):
    calls = []

    async def always_down(*a, **k):
        calls.append(1)
        raise TransientError("provider 503")

    monkeypatch.setattr(tasks, "_execute", always_down)
    run_id = uuid.uuid4().hex
    tasks.execute_agent_workflow.apply(args=[run_id, "alice", "t", "wedding"])
    assert len(calls) == 1 + settings.TASK_MAX_RETRIES                  # exactly 4 attempts, never more
    run = env.r.hgetall(run_store.run_key(run_id))
    assert run["status"] == "failed" and "retries exhausted" in run["error"]
    assert "retrying" in _events(env.r, run_id) and _events(env.r, run_id)[-1] == "done"


def test_budget_exhaustion_fails_immediately_without_retrying(env, monkeypatch):
    calls = []

    async def over_budget(*a, **k):
        calls.append(1)
        raise Exception("Budget has been exceeded! Current cost: 10.2, Max budget: 10.0")

    monkeypatch.setattr(tasks, "_execute", over_budget)
    run_id = uuid.uuid4().hex
    tasks.execute_agent_workflow.apply(args=[run_id, "alice", "t", "wedding"])
    assert len(calls) == 1
    assert env.r.hget(run_store.run_key(run_id), "error") == "budget_exceeded"


def test_unexpected_errors_are_not_retried_and_do_not_leak_details(env, monkeypatch):
    async def bug(*a, **k):
        raise KeyError("secret-internal-detail")

    monkeypatch.setattr(tasks, "_execute", bug)
    run_id = uuid.uuid4().hex
    tasks.execute_agent_workflow.apply(args=[run_id, "alice", "t", "wedding"])
    run = env.r.hgetall(run_store.run_key(run_id))
    assert run["status"] == "failed" and run["error"] == "internal_error"
    assert "secret" not in json.dumps(run)


def test_eval_task_records_scores_and_raises_regression_signal(env, monkeypatch):
    import evals

    good = {"hallucination_score": 0.95, "hallucination_passed": True, "hallucination_reason": "ok",
            "relevancy_score": 0.9, "relevancy_passed": True, "relevancy_reason": "ok"}
    bad = {**good, "hallucination_score": 0.2, "hallucination_passed": False,
           "relevancy_score": 0.3, "relevancy_passed": False}

    monkeypatch.setattr(evals, "run_metrics", lambda *a: good)
    first = tasks.evaluate_agent_run.run("r1", "p", "plan", ["ctx"], None)
    assert first["regression"] is False
    assert json.loads(env.r.hget(run_store.run_key("r1"), "eval"))["relevancy_score"] == 0.9

    monkeypatch.setattr(evals, "run_metrics", lambda *a: bad)
    last = None
    for i in range(8):                                                  # quality drops -> rolling avg sinks
        last = tasks.evaluate_agent_run.run(f"r{i + 2}", "p", "plan", ["ctx"], None)
    assert last["regression"] is True and last["rolling"]["relevancy"] < settings_relevancy()


def settings_relevancy():
    import evals
    return evals.RELEVANCY_THRESHOLD


def test_eval_failures_never_raise_into_the_pipeline(env, monkeypatch):
    import evals

    def broken(*a):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(evals, "run_metrics", broken)
    assert tasks.evaluate_agent_run.run("rX", "p", "plan", ["c"], None) is None
