from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_llm_client = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )
    return _llm_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_completion(messages: list[dict], **kwargs):
    client = get_llm_client()
    return client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=kwargs.get("temperature", settings.llm_temperature),
        max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
        tools=kwargs.get("tools"),
        tool_choice=kwargs.get("tool_choice"),
    )
