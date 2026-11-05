"""DeepEval metrics, judged by the same Claude model through the LiteLLM gateway
(so eval spend is also covered by the virtual-key budget)."""
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

import settings
from llm import get_llm

# DeepEval >= 4: every metric scores in the same direction, 1.0 = pass and `threshold` is the MINIMUM
# passing score (HallucinationMetric used to be inverted; tests/test_evals.py pins this behaviour).
HALLUCINATION_THRESHOLD = 0.8      # share of the answer that must be grounded in the specialist reports
RELEVANCY_THRESHOLD = 0.7          # share of the answer that is relevant to the request


class GatewayJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._model = get_llm()
        super().__init__(model=settings.LITELLM_MODEL if settings.LITELLM_BASE_URL else settings.DIRECT_MODEL)

    def load_model(self):
        return self._model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self._model.with_structured_output(schema).invoke(prompt)
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self._model.with_structured_output(schema).ainvoke(prompt)
        return (await self._model.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return str(self.model)


def run_metrics(user_prompt: str, plan: str, context: list[str]) -> dict:
    judge = GatewayJudge()
    case = LLMTestCase(input=user_prompt, actual_output=plan, context=[c for c in context if c])
    hallucination = HallucinationMetric(threshold=HALLUCINATION_THRESHOLD, model=judge,
                                        include_reason=True, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge,
                                      include_reason=True, async_mode=False)
    hallucination.measure(case)
    relevancy.measure(case)
    return {
        # NOTE: 1.0 = fully grounded / no hallucination, 0.0 = fully hallucinated.
        "hallucination_score": hallucination.score,
        "hallucination_passed": hallucination.is_successful(),
        "hallucination_reason": hallucination.reason,
        "relevancy_score": relevancy.score,
        "relevancy_passed": relevancy.is_successful(),
        "relevancy_reason": relevancy.reason,
    }
