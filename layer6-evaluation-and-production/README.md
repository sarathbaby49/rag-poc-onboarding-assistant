# 🚦 Layer 6 — Evaluation & Production

The onboarding assistant works on your laptop. This layer is about knowing it
works, and **keeping** it working once real people use it.

- **Lab 1 · Evaluation** — a golden set with off-topic and adversarial cases,
  code-based grading, a retrieval metric, an LLM judge, and a regression gate
  that says **SHIP** or **BLOCK** when someone changes the prompt.
- **Lab 2 · Observability** — a hand-built trace log (what LangSmith does for
  you), PII redaction before anything is written, and the p50/p95/cost/error
  dashboard.

AI safety (prompt injection, bias, privacy, when not to use an LLM) is covered
in the talk; there's no hands-on lab for it this session.

---

## Prerequisites

1. **Vector index built** — `python -m src.ingest` ("Indexed 30 chunks").
2. **LLM gateway reachable** — `python -m checks.check_llm` must pass. The gateway
   is only reachable on the company network/VPN; without it, the key-free
   exercises still work (EV1, EV2, EV4, OB1, OB2*, OB3).
   <sub>*OB2's self-check uses a fake model.</sub>
3. `pip install -r requirements.txt` (no new packages for this layer).

Optional `.env` settings (see `.env.example`):

```bash
JUDGE_MODEL=litellm_proxy/anthropic/claude-sonnet-5   # defaults to LLM_MODEL
LLM_TIMEOUT=60                                        # seconds before a call gives up
```

---

## Exercises

Full instructions: **[EXERCISES.md](EXERCISES.md)**.

| Lab | Exercise | What | File | Needs gateway? |
|-----|----------|------|------|----------------|
| 1 | **EV1** | Code-based grader | `src/eval/run_eval.py` | no |
| 1 | **EV2** | Retrieval hit-rate | `src/eval/run_eval.py` | no |
| 1 | **EV3** | LLM-as-judge (faithfulness) | `src/eval/run_eval.py` | yes (live part) |
| 1 | **EV4** | Regression gate: SHIP / BLOCK | `src/eval/run_eval.py` | no |
| 2 | **OB1** | Redact PII + secrets | `src/observability.py` | no |
| 2 | **OB2** | Trace every model call | `src/observability.py` | no (faked in check) |
| 2 | **OB3** | Dashboard numbers (stretch) | `src/observability.py` | no |

**Self-check:**

```bash
python -m checks.check_eval            # Lab 1 — EV1–EV4
python -m checks.check_observability   # Lab 2 — OB1–OB3
```

Reference solutions: `solutions/eval.py`, `solutions/observability.py` — try
first, then peek.

## The lab app

```bash
streamlit run production_lab.py
```

| Mode | Shows | Lab |
|------|-------|-----|
| **Ask** | chat with latency, tokens and cost under every answer | 2 (OB2) |
| **Eval** | run the golden set on v1/v2, per-case ✅/❌ with the judge's reasoning, then the SHIP/BLOCK gate | 1 (EV1–EV4) |
| **Traces** | p50/p95, error rate, cost per request and per prompt version, redacted log, redaction playground | 2 (OB1, OB3) |

The sidebar switch **Code under test → Reference solutions** runs the finished
version — presenters use it for the live demo (see **[DEMO.md](DEMO.md)**).

## What's new in the repo

| File | Status |
|------|--------|
| `src/assistant.py` | ✅ the system under test: prompt versions v1/v2, traced calls |
| `src/eval/run_eval.py` | 📝 Lab 1: EV1–EV4 (+ done-for-you suite runner and CLI) |
| `src/eval/golden_set.jsonl` | ✅ 8 golden + 3 off-topic + 1 adversarial case |
| `src/observability.py` | 📝 Lab 2: OB1–OB3 (+ done-for-you trace writer, cost, percentile) |
| `src/llm.py` | ✅ `complete_with_usage()` — returns tokens + latency too |
| `src/layer6_support.py` | ✅ gateway probe, solutions switch, exercise status |
| `production_lab.py` | ✅ the Streamlit lab / live demo |
| `checks/check_eval.py`, `checks/check_observability.py` | ✅ self-checks |
| `solutions/eval.py`, `solutions/observability.py` | ✅ reference answers |

Generated at runtime (git-ignored): `.traces/` (trace log), `.eval_runs/` (saved eval runs).

## Key concepts

| Concept | In one line | Where |
|---------|-------------|-------|
| Golden set | fixed questions + what a good answer must contain | `src/eval/golden_set.jsonl` |
| Code-based grader | exact, free checks (keywords, cited source, abstained?) | EV1 |
| Retrieval vs generation | score them separately so you know which to fix | EV2 vs EV3 |
| LLM-as-judge | a second model call grades faithfulness, with reasoning | EV3 |
| Regression gate | compare per case, not just averages; block on any drop | EV4 |
| Trace | one record per model call: who, what, how long, how much, ok? | OB2 |
| Redaction | strip PII/secrets *before* anything is written | OB1 |
| p95 | the slow tail — what 1 in 20 users waits | OB3 |
