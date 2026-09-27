"""LiteLLM gateway — the one place the whole app talks to a language model.

We call models through a LiteLLM proxy (an OpenAI-style gateway). That means:
  - no direct provider API key is needed — just the proxy base URL + key in .env
  - swapping models is a one-line change (LLM_MODEL in .env), code stays the same

Messages are OpenAI-style dicts:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

from __future__ import annotations

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


def ask(prompt: str, system: str | None = None, **kwargs) -> str:
    """Single-turn convenience: optional system prompt + one user message."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return complete(messages, **kwargs)
