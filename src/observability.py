"""Layer 6 — Observability  (LAB 2 — YOUR EXERCISES: OB1–OB3).

    every LLM call  ->  one trace record (who, what, how long, how much, ok?)  ->  dashboard

LangSmith (Layer 4), Langfuse and Helicone all do this for you. Here you build a
tiny version by hand so you can see inside one: a JSONL file of traces, with
personal data redacted BEFORE anything is written.

  - OB1 (core):    implement `redact`           -> strip emails, phones, cards, secrets
  - OB2 (core):    implement `traced_complete`  -> call the model + write a trace record
  - OB3 (stretch): implement `summarize`        -> p50/p95 latency, error rate, cost

Done for you: hash_user, cost_usd, write_trace, load_traces, percentile, call_llm.

Reference answers: solutions/observability.py
Self-check:        python -m checks.check_observability
Look at traces:    python -m src.observability   (or the Traces mode in production_lab.py)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone

from src import config
from src.llm import complete_with_usage

# --- OB1: redaction ------------------------------------------------------------
# (placeholder, compiled regex) pairs, applied IN ORDER. Order matters: a card
# number also looks like a phone number, so the card pattern must run first.
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # TODO(OB1): add patterns for secrets, emails, card numbers and phone numbers.
]


def redact(text: str) -> str:
    """EXERCISE OB1 — replace personal data and secrets with placeholders.

    Logs and traces get copied, shared and kept for months. Anything personal in
    them is a privacy incident waiting to happen. So redact in the app, before
    the data is written anywhere.

    Replace, in this order:
      [SECRET]  API keys / tokens:  sk-..., lsv2_..., tvly-..., ghp_..., AKIA...
      [EMAIL]   priya.s@acme.com
      [CARD]    4111 1111 1111 1111   (13–16 digits, optional spaces/dashes)
      [PHONE]   +44 7700 900123, (555) 123-4567

    Steps:
      1. Fill PII_PATTERNS above with (placeholder, re.compile(...)) pairs.
      2. For each pair, text = pattern.sub(placeholder, text). Return text.

    Hints:
      email:  r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+"
      card:   r"\\b(?:\\d[ -]?){12,15}\\d\\b"
      phone:  r"(?:\\+\\d{1,3}[\\s-]?)?\\(?\\d{3,4}\\)?[\\s-]?\\d{3}[\\s-]?\\d{3,4}\\b"
      secret: r"\\b(?:sk-[\\w-]{8,}|lsv2_[\\w-]{8,}|tvly-[\\w-]{8,}|ghp_\\w{20,}|AKIA[0-9A-Z]{16})"

    Self-check:  python -m checks.check_observability   (no key needed)
    """
    raise NotImplementedError("Exercise OB1: implement redact — see layer6 EXERCISES.md")


# --- Done for you --------------------------------------------------------------
def hash_user(user_id: str) -> str:
    """Pseudonymise a user id: you can still group by user without knowing who."""
    return "u_" + hashlib.sha256(f"onboarding-salt:{user_id}".encode()).hexdigest()[:10]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Tokens × price. Prices live in config.PRICES_PER_MILLION."""
    price_in, price_out = next(
        (p for name, p in config.PRICES_PER_MILLION.items() if name in model),
        config.DEFAULT_PRICE_PER_MILLION,
    )
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def write_trace(record: dict) -> None:
    """Append one trace record to the local trace log (.traces/traces.jsonl)."""
    config.TRACE_DIR.mkdir(exist_ok=True)
    with config.TRACE_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_traces() -> list[dict]:
    if not config.TRACE_FILE.exists():
        return []
    return [json.loads(line) for line in config.TRACE_FILE.read_text().splitlines() if line.strip()]


def clear_traces() -> None:
    if config.TRACE_FILE.exists():
        config.TRACE_FILE.unlink()


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. percentile(xs, 95) = the slow tail."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def _new_record(name: str, prompt_version: str, user_id: str) -> dict:
    return {
        "trace_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "prompt_version": prompt_version,
        "user": hash_user(user_id),
    }


# --- OB2: trace every model call -----------------------------------------------
def traced_complete(
    messages: list[dict],
    *,
    name: str,
    prompt_version: str,
    user_id: str = "anonymous",
    model: str | None = None,
    **kwargs,
) -> dict:
    """EXERCISE OB2 — call the model AND write one trace record about the call.

    Returns the dict from complete_with_usage() plus "cost_usd".

    Steps:
      1. record = _new_record(name, prompt_version, user_id)          # done-for-you
         record["input"] = redact(messages[-1]["content"])           # OB1!
      2. Time the call. In a try/except/finally:
           start = time.perf_counter()
           try:
               result = complete_with_usage(messages, model=model, **kwargs)
               result["cost_usd"] = cost_usd(result["model"], result["input_tokens"],
                                             result["output_tokens"])
               record.update(status="ok", model=result["model"],
                             input_tokens=..., output_tokens=..., cost_usd=...,
                             output=redact(result["text"]))
               return result
           except Exception as exc:
               record.update(status="error", error=f"{type(exc).__name__}: {exc}")
               raise                                  # never swallow the error
           finally:
               record["latency_s"] = round(time.perf_counter() - start, 3)
               write_trace(record)                    # ALWAYS log, even on error

    Self-check:  python -m checks.check_observability   (uses a fake model, no key needed)
    """
    raise NotImplementedError("Exercise OB2: implement traced_complete — see layer6 EXERCISES.md")


# --- OB3: the dashboard numbers ------------------------------------------------
def summarize(traces: list[dict]) -> dict:
    """EXERCISE OB3 — turn raw traces into the numbers you'd put on a dashboard.

    Return:
      {
        "requests": int,
        "error_rate": float,            # 0..1
        "p50_latency_s": float,         # use percentile(); p95 = the slow tail
        "p95_latency_s": float,
        "total_cost_usd": float,
        "cost_per_request_usd": float,
        "by_prompt_version": {"v1": {"requests": int, "cost_per_request_usd": float}, ...},
      }
    Empty input -> requests 0 and zeros everywhere (no ZeroDivisionError!).

    Self-check:  python -m checks.check_observability   (no key needed)
    """
    raise NotImplementedError("Exercise OB3: implement summarize — see layer6 EXERCISES.md")


# --- Done for you: the call every other module uses ----------------------------
def call_llm(messages: list[dict], *, name: str, prompt_version: str = "v1",
             user_id: str = "anonymous", model: str | None = None, **kwargs) -> dict:
    """Traced call if OB2 is done; otherwise a plain (untraced) call, so the
    eval exercises still work before you've finished OB2."""
    try:
        return traced_complete(messages, name=name, prompt_version=prompt_version,
                               user_id=user_id, model=model, **kwargs)
    except NotImplementedError:
        result = complete_with_usage(messages, model=model, **kwargs)
        result["cost_usd"] = cost_usd(result["model"], result["input_tokens"], result["output_tokens"])
        return result


def _use_solutions_if_asked(names: tuple[str, ...], solution_module: str) -> None:
    """`--solutions` on the CLI: run the reference answers (presenter demo)."""
    if "--solutions" not in sys.argv:
        return
    import importlib

    from src.layer6_support import use_reference_solutions

    use_reference_solutions()
    ref = importlib.import_module(solution_module)
    globals().update({name: getattr(ref, name) for name in names})


def _cli() -> None:
    _use_solutions_if_asked(("PII_PATTERNS", "redact", "traced_complete", "summarize"),
                            "solutions.observability")
    traces = load_traces()
    print(f"{len(traces)} traces in {config.TRACE_FILE.relative_to(config.BASE_DIR)}\n")
    for t in traces[-5:]:
        print(f"[{t.get('status')}] {t.get('name')} {t.get('prompt_version')} "
              f"{t.get('latency_s')}s ${t.get('cost_usd', 0):.5f}  {str(t.get('input'))[:60]!r}")
    try:
        print("\n" + json.dumps(summarize(traces), indent=2))
    except NotImplementedError as exc:
        print(f"\n(summary: {exc})")


if __name__ == "__main__":
    if "--clear" in sys.argv:
        clear_traces()
        print("Trace log cleared.")
    else:
        _cli()
