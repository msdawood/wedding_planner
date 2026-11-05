"""Error classification shared by the graph and the Celery tasks."""


class BudgetExceeded(Exception):
    """LiteLLM refused the call because the virtual key's budget is spent. Never retry."""


class TransientError(Exception):
    """Worth retrying with backoff (network blips, provider 5xx, proxy rate limits)."""


def is_budget_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, BudgetExceeded) or ("budget" in text and ("exceeded" in text or "over" in text))


def is_transient_error(exc: BaseException) -> bool:
    """Classify without importing provider SDKs at module import time."""
    if is_budget_error(exc):
        return False
    names = {c.__name__ for c in type(exc).__mro__}
    transient = {
        "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
        "ServiceUnavailableError", "OverloadedError", "ConnectError", "ReadTimeout",
        "ConnectTimeout", "TimeoutError", "ConnectionError", "TransientError",
    }
    return bool(names & transient)
