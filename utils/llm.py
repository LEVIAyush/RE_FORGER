import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set.\n"
        "Copy .env.example to .env and add your free key from "
        "https://console.groq.com/keys"
    )

client = Groq(api_key=_api_key)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def call_llm(messages: list, tools: list = None, temperature: float = 0.1, max_retries: int = 2):
    """
    Thin wrapper around the Groq chat completions endpoint.

    Fixes vs. the previous version:
    - `tools` / `tool_choice` are only attached when tools are actually
      provided. Sending `tool_choice` with no `tools` array is invalid per
      the OpenAI-compatible spec (some backends tolerate it, but it's not
      something to rely on).
    - Transient errors (network blips, rate limits) are retried a couple of
      times with a short backoff before raising, instead of crashing the
      whole agent run on the first hiccup.
    """
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempt(s): {last_error}")
