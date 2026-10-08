"""REFERENCE SOLUTION for src/observability.py — Lab 2, Exercises OB1–OB3.

Try each yourself first! If you're stuck or out of time, copy the relevant
function body (and PII_PATTERNS) into src/observability.py.

  - OB1 -> redact            (strip personal data + secrets before logging)
  - OB2 -> traced_complete   (call the model + always write a trace record)
  - OB3 -> summarize         (p50/p95 latency, error rate, cost)
"""

from __future__ import annotations

import re
import time

from src.llm import complete_with_usage
from src.observability import _new_record, cost_usd, percentile, write_trace

# --- OB1 ------------------------------------------------------------------------
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[SECRET]", re.compile(r"\b(?:sk-[\w-]{8,}|lsv2_[\w-]{8,}|tvly-[\w-]{8,}|ghp_\w{20,}|AKIA[0-9A-Z]{16})")),
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){12,15}\d\b")),   # before PHONE: cards look like phones
    ("[PHONE]", re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{3,4}\)?[\s-]?\d{3}[\s-]?\d{3,4}\b")),
]


def redact(text: str) -> str:
    for placeholder, pattern in PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


# --- OB2 ------------------------------------------------------------------------
def traced_complete(
    messages: list[dict],
    *,
    name: str,
    prompt_version: str,
    user_id: str = "anonymous",
    model: str | None = None,
    **kwargs,
) -> dict:
    record = _new_record(name, prompt_version, user_id)
    record["input"] = redact(messages[-1]["content"])
    start = time.perf_counter()
    try:
        result = complete_with_usage(messages, model=model, **kwargs)
        result["cost_usd"] = cost_usd(result["model"], result["input_tokens"], result["output_tokens"])
        record.update(
            status="ok",
            model=result["model"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=round(result["cost_usd"], 6),
            output=redact(result["text"]),
        )
        return result
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["latency_s"] = round(time.perf_counter() - start, 3)
        write_trace(record)


# --- OB3 ------------------------------------------------------------------------
def summarize(traces: list[dict]) -> dict:
    n = len(traces)
    latencies = [t.get("latency_s", 0.0) for t in traces]
    costs = [t.get("cost_usd", 0.0) for t in traces]
    by_version: dict[str, dict] = {}
    for t in traces:
        v = by_version.setdefault(t.get("prompt_version", "?"), {"requests": 0, "cost": 0.0})
        v["requests"] += 1
        v["cost"] += t.get("cost_usd", 0.0)
    return {
        "requests": n,
        "error_rate": sum(t.get("status") == "error" for t in traces) / n if n else 0.0,
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "total_cost_usd": sum(costs),
        "cost_per_request_usd": sum(costs) / n if n else 0.0,
        "by_prompt_version": {
            version: {"requests": v["requests"], "cost_per_request_usd": v["cost"] / v["requests"]}
            for version, v in by_version.items()
        },
    }
