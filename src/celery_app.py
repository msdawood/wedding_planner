"""Celery application. Kept free of heavy imports so the API container can import it
(to enqueue by task name) without loading LangGraph / DeepEval."""
import os
import sys

# `celery -A tasks.celery_app` only has this folder on sys.path while it loads the app, then removes
# it. Our tasks import modules lazily (wedding_graph, evals), so pin the folder permanently.
_HERE = os.path.dirname(os.path.abspath(__file__))
# Unconditional on purpose: Celery's own temporary entry for this same folder may already be present,
# and it removes the *first* match on exit, so we need a second copy that survives.
sys.path.append(_HERE)

from celery import Celery  # noqa: E402

import settings  # noqa: E402

RUN_TASK = "tasks.execute_agent_workflow"
EVAL_TASK = "tasks.evaluate_agent_run"

celery_app = Celery("agent_workers", broker=settings.REDIS_URL, include=["tasks"])

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,                 # results live in our Redis run store, not a Celery backend
    broker_connection_retry_on_startup=True,

    # --- durability: ack only after the task finishes; requeue if the worker process dies ---
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Must exceed the hard time limit, otherwise Redis redelivers a task that is still running.
    broker_transport_options={"visibility_timeout": settings.TASK_TIME_LIMIT + 300},

    # --- bounds ---
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.TASK_TIME_LIMIT,
    worker_max_tasks_per_child=settings.WORKER_MAX_TASKS_PER_CHILD,
    worker_max_memory_per_child=settings.WORKER_MAX_MEMORY_PER_CHILD_KB,

    # --- queues: evaluations never compete with user-facing runs ---
    task_default_queue="runs",
    task_routes={EVAL_TASK: {"queue": "evals"}},
)
