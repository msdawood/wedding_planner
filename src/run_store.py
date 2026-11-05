"""Run tracking in Redis, shared by the API (async client) and the worker (sync client).

Why not plain Pub/Sub: a message published before the browser connects is lost, so a late
subscriber would wait forever. Instead every event is appended to a Redis *Stream* and the
run's status lives in a hash, so /stream can replay from the beginning (or from Last-Event-ID)
and /runs/{id} always answers.

Keys
  run:{run_id}          hash   user_id, thread_id, status, result(json), error, trace_id, eval(json), ...
  run:{run_id}:events   stream type, data(json)
  rl:{user_id}:{minute} int    per-user rate limit counter
  eval:{metric}         list   rolling window of recent eval scores
"""
import json
import time

import settings

QUEUED, RUNNING, RETRYING = "queued", "running", "retrying"
SUCCEEDED, FAILED, REFUSED, NEEDS_INFO = "succeeded", "failed", "refused", "needs_info"
TERMINAL = {SUCCEEDED, FAILED, REFUSED, NEEDS_INFO}


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


def events_key(run_id: str) -> str:
    return f"run:{run_id}:events"


def _now() -> str:
    return str(time.time())


def _encode(fields: dict) -> dict:
    return {k: v if isinstance(v, (str, int, float)) else json.dumps(v, default=str) for k, v in fields.items()}


# ---------------------------------------------------------------- sync (Celery worker)
def set_status(r, run_id: str, status: str, **fields) -> None:
    r.hset(run_key(run_id), mapping=_encode({"status": status, "updated_at": _now(), **fields}))
    r.expire(run_key(run_id), settings.RUN_TTL_SECONDS)


def add_event(r, run_id: str, type_: str, data: dict | None = None) -> None:
    r.xadd(events_key(run_id), {"type": type_, "data": json.dumps(data or {}, default=str)},
           maxlen=1000, approximate=True)
    r.expire(events_key(run_id), settings.RUN_TTL_SECONDS)


def record_eval_scores(r, scores: dict[str, float], window: int = 20) -> dict[str, float]:
    """Push scores into rolling windows; return the moving average per metric."""
    averages = {}
    for metric, value in scores.items():
        key = f"eval:{metric}"
        r.lpush(key, value)
        r.ltrim(key, 0, window - 1)
        vals = [float(v) for v in r.lrange(key, 0, window - 1)]
        averages[metric] = sum(vals) / len(vals)
    return averages


# ---------------------------------------------------------------- async (FastAPI)
async def a_create_run(r, run_id: str, user_id: str, thread_id: str) -> None:
    await r.hset(run_key(run_id), mapping={
        "user_id": user_id, "thread_id": thread_id, "status": QUEUED,
        "created_at": _now(), "updated_at": _now()})
    await r.expire(run_key(run_id), settings.RUN_TTL_SECONDS)
    await r.xadd(events_key(run_id), {"type": "queued", "data": "{}"}, maxlen=1000, approximate=True)
    await r.expire(events_key(run_id), settings.RUN_TTL_SECONDS)


async def a_get_run(r, run_id: str) -> dict | None:
    data = await r.hgetall(run_key(run_id))
    if not data:
        return None
    for field in ("result", "eval"):
        if field in data:
            try:
                data[field] = json.loads(data[field])
            except json.JSONDecodeError:
                pass
    return data


async def a_fail(r, run_id: str, error: str) -> None:
    await r.hset(run_key(run_id), mapping={"status": FAILED, "error": error, "updated_at": _now()})
    await r.xadd(events_key(run_id), {"type": "done", "data": json.dumps({"status": FAILED, "error": error})})


async def a_rate_limited(r, user_id: str) -> bool:
    """Fixed-window limiter: RUNS_PER_MINUTE runs per user per minute."""
    key = f"rl:{user_id}:{int(time.time() // 60)}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, 70)
    return count > settings.RUNS_PER_MINUTE
