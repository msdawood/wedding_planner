"""Locks the DeepEval semantics we depend on, using a stub judge (no network, no cost)."""
import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric   # noqa: E402
from deepeval.models import DeepEvalBaseLLM                                 # noqa: E402
from deepeval.test_case import LLMTestCase                                  # noqa: E402

import evals                                                                # noqa: E402


class StubJudge(DeepEvalBaseLLM):
    def __init__(self, verdict: str):
        self.verdict = verdict
        super().__init__(model="stub")

    def load_model(self):
        return None

    def generate(self, prompt, schema=None):
        if schema is None:
            return "{}"
        name = schema.__name__
        if name == "Verdicts":
            return schema(verdicts=[{"verdict": self.verdict, "reason": "stub"}])
        if name == "Statements":
            return schema(statements=["The Paris venue costs 5000"])
        if name.endswith("Reason"):
            return schema(reason="stub reason")
        raise AssertionError(f"unexpected schema {name}")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "stub"


CASE = LLMTestCase(input="plan a wedding", actual_output="The Paris venue costs 5000",
                   context=["The Paris venue costs 5000"])


def _hallucination(verdict):
    m = HallucinationMetric(threshold=evals.HALLUCINATION_THRESHOLD, model=StubJudge(verdict),
                            include_reason=True, async_mode=False)
    m.measure(CASE)
    return m


def test_grounded_answer_scores_high_and_passes():
    m = _hallucination("yes")
    assert m.score == 1.0 and m.is_successful()


def test_hallucinated_answer_scores_low_and_fails():
    m = _hallucination("no")
    assert m.score < evals.HALLUCINATION_THRESHOLD and not m.is_successful()


def test_relevancy_api():
    m = AnswerRelevancyMetric(threshold=evals.RELEVANCY_THRESHOLD, model=StubJudge("yes"),
                              include_reason=True, async_mode=False)
    m.measure(CASE)
    assert m.score == 1.0 and m.is_successful() and m.reason


def test_run_metrics_with_gateway_judge(monkeypatch):
    """Our GatewayJudge adapter plugs into DeepEval end to end (structured output -> schema)."""
    class FakeChat:
        def with_structured_output(self, schema):
            outer = self
            class S:
                def invoke(self, prompt):
                    n = schema.__name__
                    if n == "Verdicts":
                        return schema(verdicts=[{"verdict": "yes", "reason": "r"}])
                    if n == "Statements":
                        return schema(statements=["s"])
                    return schema(reason="why")
                async def ainvoke(self, prompt):
                    return self.invoke(prompt)
            return S()

        def invoke(self, prompt):
            class M:
                content = "{}"
            return M()

    monkeypatch.setattr(evals, "get_llm", lambda *a, **k: FakeChat())
    res = evals.run_metrics("plan a wedding", "The Paris venue costs 5000", ["The Paris venue costs 5000"])
    assert res["hallucination_score"] == 1.0 and res["hallucination_passed"] is True
    assert res["relevancy_score"] == 1.0 and res["relevancy_passed"] is True
