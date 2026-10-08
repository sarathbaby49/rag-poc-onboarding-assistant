"""Self-check for Lab 2 — the Observability exercises (OB1–OB3).

Key-free: OB1 redaction, OB2 with a fake model call (incl. the error path),
OB3 dashboard numbers on hand-made traces. Your real trace log is never touched.

Run from the repo root:  python -m checks.check_observability
(Presenters: `--solutions` checks the reference answers.)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from src import layer6_support

if "--solutions" in sys.argv:
    layer6_support.use_reference_solutions()

from src import config, observability as obs  # noqa: E402


def _check(label: str, cond) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return bool(cond)


def _todo(code: str) -> bool:
    print(f"❌ {code} still a TODO — see layer6-evaluation-and-production/EXERCISES.md")
    return False


def main() -> None:
    results: list[bool] = []

    print("-- Implementation status --")
    for code, name, done in layer6_support.exercise_status():
        if code.startswith("OB"):
            print(f"{'✅' if done else '❌'} {code} {name}() — {'implemented' if done else 'NOT implemented yet'}")

    print("\n-- OB1: redact (no key) --")
    try:
        r = obs.redact
        results.append(_check("email -> [EMAIL]", r("mail priya.s@acme.com now") == "mail [EMAIL] now"))
        phone = r("call +44 7700 900123")
        results.append(_check("phone -> [PHONE]", "[PHONE]" in phone and "7700" not in phone))
        card = r("card 4111 1111 1111 1111 charged")
        results.append(_check("card -> [CARD] (not [PHONE])", "[CARD]" in card and "4111" not in card))
        results.append(_check("API key -> [SECRET]", r("key sk-live-abc123def456") == "key [SECRET]"))
        plain = "Run make run on 2026-10-07, then open PR #42 for ACME-12."
        results.append(_check("leaves normal text, dates and ticket ids alone", r(plain) == plain))
    except NotImplementedError:
        results.append(_todo("OB1"))

    print("\n-- OB2: traced_complete (fake model, no key) --")
    tmp = Path(tempfile.mkdtemp())
    home = sys.modules[obs.traced_complete.__module__]   # src or solutions
    real_dir, real_file, real_call = config.TRACE_DIR, config.TRACE_FILE, home.complete_with_usage
    config.TRACE_DIR, config.TRACE_FILE = tmp, tmp / "traces.jsonl"
    try:
        home.complete_with_usage = lambda messages, model=None, **kw: {
            "text": "Ping priya.s@acme.com", "model": "claude-sonnet-5",
            "input_tokens": 1000, "output_tokens": 100, "latency_s": 0.01}
        out = obs.traced_complete([{"role": "user", "content": "I'm tom@northwind.io"}],
                                  name="test", prompt_version="v9", user_id="tom")
        traces = obs.load_traces()
        t = traces[-1] if traces else {}
        results.append(_check("returns the model result with cost_usd", "cost_usd" in out and out["text"]))
        results.append(_check("writes one trace record", len(traces) == 1))
        results.append(_check("record has status, latency, tokens, cost, prompt_version",
                              t.get("status") == "ok" and "latency_s" in t and t.get("input_tokens") == 1000
                              and abs(t.get("cost_usd", 0) - 0.003) < 1e-6 and t.get("prompt_version") == "v9"))
        results.append(_check("input AND output redacted, user id hashed",
                              "@" not in t.get("input", "@") and "@" not in t.get("output", "@")
                              and t.get("user", "tom") != "tom"))

        def boom(*a, **k):
            raise TimeoutError("gateway timed out")
        home.complete_with_usage = boom
        try:
            obs.traced_complete([{"role": "user", "content": "hi"}], name="test", prompt_version="v9")
            raised = False
        except TimeoutError:
            raised = True
        last = obs.load_traces()[-1]
        results.append(_check("on error: re-raises AND still logs status='error'",
                              raised and last.get("status") == "error"))
    except NotImplementedError:
        results.append(_todo("OB2"))
    finally:
        config.TRACE_DIR, config.TRACE_FILE, home.complete_with_usage = real_dir, real_file, real_call

    print("\n-- OB3: summarize (no key) --")
    try:
        fake = [{"latency_s": s, "cost_usd": 0.01, "status": "ok", "prompt_version": "v1"} for s in range(1, 20)]
        fake.append({"latency_s": 30.0, "cost_usd": 0.05, "status": "error", "prompt_version": "v2"})
        s = obs.summarize(fake)
        results.append(_check("counts requests and error rate", s["requests"] == 20 and abs(s["error_rate"] - 0.05) < 1e-9))
        results.append(_check("p50 ≈ 10s, p95 well above it (the slow tail)",
                              9 <= s["p50_latency_s"] <= 11 and s["p95_latency_s"] >= 19))
        results.append(_check("total and per-request cost",
                              abs(s["total_cost_usd"] - 0.24) < 1e-9 and abs(s["cost_per_request_usd"] - 0.012) < 1e-9))
        results.append(_check("breaks down by prompt version",
                              s["by_prompt_version"]["v2"]["requests"] == 1))
        empty = obs.summarize([])
        results.append(_check("empty trace list -> zeros, no crash", empty["requests"] == 0))
    except NotImplementedError:
        print("⚠️  OB3 (stretch) not implemented yet — optional")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} key-free checks passed (OB3 is optional).")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
