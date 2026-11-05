"""Model factory.

All LLM traffic goes through the LiteLLM proxy when LITELLM_BASE_URL is set, so the
proxy's virtual key (budget + rate limits) is the single spending choke point.
Without it (plain `python wedding_graph.py` on a laptop) we fall back to calling
Anthropic directly, exactly like before.
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

import settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if settings.LITELLM_BASE_URL:
        
        # LiteLLM exposes an OpenAI-compatible API and translates to Anthropic.
        return ChatOpenAI(
            model=settings.LITELLM_MODEL,
            base_url=settings.LITELLM_BASE_URL.rstrip("/") + "/v1",
            api_key=settings.LITELLM_API_KEY or "unused",
            temperature=temperature,
            max_retries=2,
            timeout=120,
        )
    return init_chat_model(settings.DIRECT_MODEL, model_provider="anthropic", temperature=temperature)
