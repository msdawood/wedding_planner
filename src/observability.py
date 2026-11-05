"""Langfuse tracing.

The Langfuse Python SDK (v4) is built on OpenTelemetry: every LangChain/LangGraph run becomes an
OTel trace whose spans are exported to Langfuse's OTLP endpoint (/api/public/otel). SDK v4 needs a
Langfuse *server* >= 3.63 (the v3 stack in docker-compose.yml). Do NOT upgrade the SDK to v5
without also moving the server to Langfuse v4 (>= 4.59): v5 does not work against server v3.

Tracing must never break a run, so every call here degrades to a no-op on failure.
"""
from langfuse import get_client
from langfuse.langchain import CallbackHandler

import logging

import settings

log = logging.getLogger(__name__)


def trace_id_for(run_id: str) -> str | None:
    """Deterministic trace id per run: retries and crash-recovery attempts land on the same
    trace, and the eval job can attach scores without being handed the id."""
    if not settings.LANGFUSE_ENABLED:
        return None
    try:
        return get_client().create_trace_id(seed=run_id)
    except Exception:                               # noqa: BLE001
        log.exception("Could not derive Langfuse trace id")
        return None


def new_handler(run_id: str):
    """A LangChain callback handler bound to this run's trace."""
    if not settings.LANGFUSE_ENABLED:
        return None
    try:
        trace_id = trace_id_for(run_id)
        return CallbackHandler(trace_context={"trace_id": trace_id}) if trace_id else CallbackHandler()
    except Exception:                               # noqa: BLE001
        log.exception("Langfuse handler unavailable; continuing without tracing")
        return None


def trace_metadata(user_id: str, thread_id: str, run_id: str) -> dict:
    # Keys understood by the Langfuse LangChain integration.
    return {
        "langfuse_session_id": thread_id,
        "langfuse_user_id": user_id,
        "langfuse_tags": ["wedding_planner"],
        "run_id": run_id,
    }


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    if not (settings.LANGFUSE_ENABLED and trace_id):
        return
    try:
        get_client().create_score(trace_id=trace_id, name=name, value=float(value),
                                  data_type="NUMERIC", comment=(comment or "")[:500])
    except Exception:                               # noqa: BLE001
        log.exception("Could not write Langfuse score %s", name)


def flush() -> None:
    if not settings.LANGFUSE_ENABLED:
        return
    try:
        get_client().flush()
    except Exception:                               # noqa: BLE001
        log.exception("Langfuse flush failed")
