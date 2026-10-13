"""Provider-agnostic LLM access.

The baseline (call_llm) and the agent (get_chat_model) read the SAME settings, so both
always use the same provider and model -- this keeps the evaluation comparison fair.

Environment variables:
  LLM_PROVIDER     gemini (default) | groq | ollama
  LLM_MODEL        model ID; defaults below if unset
  LLM_TEMPERATURE  default 0.1
  LLM_RPM          max requests per minute, enforced client-side (defaults below; 0 = no limit).
                   Gemini's free tier limits requests per minute AND per day, per model.
  GOOGLE_API_KEY   for gemini (GEMINI_API_KEY is accepted as a fallback)
  GROQ_API_KEY     for groq
  OLLAMA_URL       for ollama, default http://localhost:11434
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",  # free tier: gemini-3.6-flash allows only 20 requests/day
    "groq": "llama-3.3-70b-versatile",
    "ollama": "phi3:mini",
}
DEFAULT_RPM = {"gemini": 10, "groq": 30, "ollama": 0}

_rate_limiter = None


def get_rate_limiter(provider):
    """One limiter shared by every model instance, so all agent calls are spaced out together."""
    global _rate_limiter
    rpm = float(os.getenv("LLM_RPM", DEFAULT_RPM[provider]))
    if rpm <= 0:
        return None
    if _rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _rate_limiter = InMemoryRateLimiter(requests_per_second=rpm / 60,
                                            check_every_n_seconds=0.2, max_bucket_size=1)
    return _rate_limiter


def get_settings():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use gemini, groq or ollama.")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.1))
    return provider, model, temperature


def get_chat_model():
    """Return a LangChain chat model for the configured provider (used by the agent)."""
    provider, model, temperature = get_settings()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY to use LLM_PROVIDER=gemini.")
        # max_retries rides out free-tier rate limits (HTTP 429).
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      google_api_key=api_key, max_retries=6,
                                      rate_limiter=get_rate_limiter(provider))

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY to use LLM_PROVIDER=groq.")
        return ChatGroq(model=model, temperature=temperature, max_retries=6,
                        rate_limiter=get_rate_limiter(provider))

    # ollama: optional, only for people running a local model.
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise RuntimeError("LLM_PROVIDER=ollama needs `pip install langchain-ollama` "
                           "(not in requirements.txt on purpose).") from e
    return ChatOllama(model=model, temperature=temperature,
                      base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"))


def message_text(message):
    """Plain text of a chat model reply. Some models (e.g. Gemini 3) return a list of parts."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def _call_ollama(prompt, model, temperature):
    # Plain REST call so the baseline needs no extra package for Ollama.
    response = requests.post(
        f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True,
              "options": {"temperature": temperature}},
        stream=True,
        timeout=300
    )

    output = []

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))
        token = data.get("response", "")
        output.append(token)

        if data.get("done", False):
            break

    return "".join(output)


def call_llm(prompt: str) -> str:
    """Single prompt -> text. Used by the baseline chatbot."""
    provider, model, temperature = get_settings()
    if provider == "ollama":
        return _call_ollama(prompt, model, temperature)
    return message_text(get_chat_model().invoke(prompt))
