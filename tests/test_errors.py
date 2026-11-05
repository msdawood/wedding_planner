import httpx
import openai
import pytest

from errors import BudgetExceeded, TransientError, is_budget_error, is_transient_error

REQ = httpx.Request("POST", "http://litellm_proxy:4000/v1/chat/completions")


def _status_error(code, message, err_type):
    body = {"error": {"message": message, "type": err_type, "code": str(code)}}
    resp = httpx.Response(code, request=REQ, json=body)
    cls = {400: openai.BadRequestError, 429: openai.RateLimitError, 500: openai.InternalServerError}[code]
    return cls(message, response=resp, body=body["error"])


@pytest.mark.parametrize("message", [
    # exact strings from litellm/proxy/auth/auth_checks.py
    "Budget has been exceeded! Key=celery-worker Current cost: 10.02, Max budget: 10.0",
    "ExceededBudget: User=abc over budget. Spend=3.2, Budget=3.0",
])
def test_litellm_budget_errors_are_recognised_through_the_openai_client(message):
    exc = _status_error(400, message, "budget_exceeded")
    assert is_budget_error(exc) and not is_transient_error(exc)
    assert is_budget_error(BudgetExceeded("x"))


def test_rate_limits_and_provider_errors_are_transient_not_budget():
    rl = _status_error(429, "Max parallel request limit reached", "rate_limit_error")
    assert is_transient_error(rl) and not is_budget_error(rl)
    assert is_transient_error(_status_error(500, "upstream exploded", "api_error"))
    assert is_transient_error(openai.APIConnectionError(request=REQ))
    assert is_transient_error(TransientError("x"))


def test_ordinary_bugs_are_neither():
    for exc in (KeyError("x"), ValueError("bad"), RuntimeError("boom")):
        assert not is_transient_error(exc) and not is_budget_error(exc)
