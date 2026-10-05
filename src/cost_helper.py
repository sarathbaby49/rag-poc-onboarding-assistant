"""Done-for-you plumbing for the Layer 5 exercises and the Cost Lab.

You don't edit this file. It:
  - calls a model through the gateway and keeps the token usage
    (src/llm.complete() only returns text — cost needs the usage too)
  - normalizes usage from LiteLLM and LangChain into the dict src/cost.py expects
  - prices calls with YOUR cost_of (C1), returning None until C1 is done
  - runs the Layer 4 onboarding-plan graph with YOUR model_for_step (C3)
    choosing the model at every step, and records what each step cost

The graph runs from solutions/graph.py, so C3 works even if the Layer 4 graph
exercises in src/graph.py aren't finished.
"""

from __future__ import annotations

import time
from typing import Any

from src import config


# --- usage normalization ------------------------------------------------------
def _get(obj: Any, key: str, default: Any = 0) -> Any:
    """Read a field from either a dict or an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_usage(usage: Any) -> dict:
    """Turn LiteLLM `response.usage` or LangChain `usage_metadata` into:

        {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"}

    where input_tokens are the FRESH (uncached) input tokens only. Both formats
    report a total input count that includes cached tokens, so we subtract them.
    """
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}

    # LangChain usage_metadata: input_tokens / output_tokens / input_token_details
    if _get(usage, "input_token_details", None) is not None or (
        isinstance(usage, dict) and "input_tokens" in usage
    ):
        details = _get(usage, "input_token_details", {}) or {}
        total_in = _get(usage, "input_tokens", 0) or 0
        read = _get(details, "cache_read", 0) or 0
        write = _get(details, "cache_creation", 0) or 0
        out = _get(usage, "output_tokens", 0) or 0
    else:
        # LiteLLM / OpenAI style: prompt_tokens / completion_tokens (+ cache fields)
        total_in = _get(usage, "prompt_tokens", 0) or 0
        out = _get(usage, "completion_tokens", 0) or 0
        read = _get(usage, "cache_read_input_tokens", 0) or _get(
            _get(usage, "prompt_tokens_details", None), "cached_tokens", 0
        ) or 0
        write = _get(usage, "cache_creation_input_tokens", 0) or 0

    return {
        "input_tokens": max(int(total_in) - int(read) - int(write), 0),
        "output_tokens": int(out),
        "cache_read_tokens": int(read),
        "cache_write_tokens": int(write),
    }


# --- costing --------------------------------------------------------------------
def safe_cost(usage: dict, model: str) -> tuple[float | None, str]:
    """Price a call with the student's cost_of. Returns (cost or None, note)."""
    from src import cost

    try:
        return cost.cost_of(usage, model), ""
    except NotImplementedError:
        return None, "finish C1 (src/cost.py) to see costs"
    except KeyError as exc:
        return None, str(exc)


def short_name(model: str) -> str:
    """'litellm_proxy/anthropic/claude-haiku-4-5' -> 'claude-haiku-4-5'."""
    return model.rsplit("/", 1)[-1]


# --- direct model calls (Compare / Router / Caching modes) ----------------------
def call(messages: list[dict], model: str, max_tokens: int = 600) -> dict:
    """Call a model through the gateway; return text, usage, cost and latency."""
    import litellm

    litellm.drop_params = True
    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=messages,
        api_base=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    usage = normalize_usage(getattr(response, "usage", None))
    cost_usd, note = safe_cost(usage, model)
    try:  # LiteLLM's own estimate — a cross-check for your C1
        gateway_cost = float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001 - unknown/new models have no LiteLLM price yet
        gateway_cost = None
    return {
        "text": response.choices[0].message.content or "",
        "model": model,
        "usage": usage,
        "cost": cost_usd,
        "cost_note": note,
        "litellm_cost": gateway_cost,
        "latency_s": latency,
    }


def rag_messages(question: str, k: int = config.TOP_K) -> tuple[list[dict], list[dict]]:
    """A plain grounded-answer prompt (retrieve -> context -> question)."""
    from src.cost import SYSTEM_PROMPT, format_hits
    from src.rag_helper import semantic_search

    hits = semantic_search(question, k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{format_hits(hits)}\n\nQuestion: {question}"},
    ]
    return messages, hits


# --- plan graph with a model per step (C3) --------------------------------------
_STATE: dict[str, Any] = {"step": None, "attempts": 0, "ledger": None, "installed": False}


def _routed_llm_call(system: str, user: str) -> str:
    """Replacement for solutions.graph._llm_call that asks model_for_step (C3)."""
    from langchain_openai import ChatOpenAI

    from src import cost

    step = _STATE["step"] or "draft_plan"
    try:
        model = cost.model_for_step(step, _STATE["attempts"])
    except NotImplementedError:
        model = config.STRONG_MODEL
    llm = ChatOpenAI(
        model=model,
        base_url=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        max_tokens=config.MAX_TOKENS,
    )
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    usage = normalize_usage(getattr(resp, "usage_metadata", None))
    cost_usd, _ = safe_cost(usage, model)
    if _STATE["ledger"] is not None:
        _STATE["ledger"].append({"step": step, "attempt": _STATE["attempts"], "model": model, **usage, "cost": cost_usd})
    return resp.content


def _install_graph_routing():
    """Patch solutions.graph once so each model call knows which step it serves."""
    import solutions.graph as sg

    if _STATE["installed"]:
        return sg

    def with_step(step, fn):
        def wrapper(*args, **kwargs):
            _STATE["step"] = step
            try:
                return fn(*args, **kwargs)
            finally:
                _STATE["step"] = None
        return wrapper

    orig_replan_node = sg.replan

    def replan_node(state):
        _STATE["attempts"] = state.get("attempts", 0)
        return orig_replan_node(state)

    sg._parse_request = with_step("parse_request", sg._parse_request)
    sg._llm_plan = with_step("draft_plan", sg._llm_plan)
    sg._llm_replan = with_step("replan", sg._llm_replan)
    sg.replan = replan_node            # build_graph() picks this up on each turn
    sg._llm_call = _routed_llm_call
    _STATE["installed"] = True
    return sg


def routed_graph_turn(user_message: str, graph_state, thread_id: str, checkpointer, ledger: list):
    """One turn of the plan bot with per-step models; appends one row per model call."""
    sg = _install_graph_routing()
    _STATE["ledger"] = ledger
    try:
        return sg.run_graph_turn(
            user_message, graph_state=graph_state, thread_id=thread_id, checkpointer=checkpointer
        )
    finally:
        _STATE["ledger"] = None
