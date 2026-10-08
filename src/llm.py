"""LiteLLM gateway — the one place the whole app talks to a language model.

We call models through a LiteLLM proxy (an OpenAI-style gateway). That means:
  - no direct provider API key is needed — just the proxy base URL + key in .env
  - swapping models is a one-line change (LLM_MODEL in .env), code stays the same

Messages are OpenAI-style dicts:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

from __future__ import annotations

import time

import litellm

from src import config

# Silently drop params a given model doesn't support (e.g. temperature) so the
# same call works across providers behind the proxy.
litellm.drop_params = True


def complete(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """Send chat messages through the gateway and return the reply text."""
    response = litellm.completion(
        model=model or config.LLM_MODEL,
        messages=messages,
        api_base=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def complete_with_usage(messages: list[dict], model: str | None = None, **kwargs) -> dict:
    """Like complete(), but also returns what observability needs (Layer 6).

    Returns {"text", "model", "input_tokens", "output_tokens", "latency_s"}.
    """
    kwargs.setdefault("timeout", config.LLM_TIMEOUT)
    start = time.perf_counter()
    response = litellm.completion(
        model=model or config.LLM_MODEL,
        messages=messages,
        api_base=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    return {
        "text": response.choices[0].message.content or "",
        "model": model or config.LLM_MODEL,
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "latency_s": time.perf_counter() - start,
    }


def ask(prompt: str, system: str | None = None, **kwargs) -> str:
    """Single-turn convenience: optional system prompt + one user message."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return complete(messages, **kwargs)
