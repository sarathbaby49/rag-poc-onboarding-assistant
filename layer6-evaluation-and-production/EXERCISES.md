# 🚦 Exercises — Evaluation & Production (Layer 6)

Two hands-on labs for the Layer 6 session: **Lab 1** adds an eval suite with a
regression gate, **Lab 2** adds tracing with PII redaction.

Difficulty: 🟢 easy · 🟡 medium · 🔵 explore.

---

## Before you start

- [ ] `git checkout layer-6 && git pull`
- [ ] `python -m src.ingest` done ("Indexed 30 chunks")
- [ ] `python -m checks.check_llm` passes (needs the company network / VPN)
- [ ] `streamlit run production_lab.py` opens

**Files you edit:** `src/eval/run_eval.py` (Lab 1), `src/observability.py` (Lab 2).

**Self-check:**

```bash
python -m checks.check_eval            # Lab 1 — EV1–EV4
python -m checks.check_observability   # Lab 2 — OB1–OB3
```

Both start at **0 passed** with "still a TODO" lines — that's expected. Most
checks need **no gateway**; live parts are skipped (not failed) when it's
unreachable. Stuck or out of time? Reference answers are in `solutions/`.

---

## ⏱️ Run of show (hands-on blocks inside the 90-min session)

| When (in the session) | What |
|------|------|
| after "LLM-as-judge" (~0:25) | **Demo** — Eval mode: run v1, read a judge verdict |
| **Lab 1** (~0:28–0:43) | **EV1** grader · **EV2** retrieval hit · **EV4** gate · (**EV3** judge if the gateway is up) |
| end of Lab 1 | **Demo** — run v2, compare → **BLOCK** |
| **Lab 2** (~0:57–0:67) | **OB1** redact · **OB2** trace → Ask mode → Traces mode · (**OB3** stretch) |

---

## Lab 1 — Evaluation

The system under test is `ask_assistant(question, prompt_version)` in
`src/assistant.py` (done for you). Two prompt versions ship:

- **v1** — production: answer only from context; abstain when retrieval finds nothing relevant.
- **v2** — a "friendlier" prompt a PM wants because users complained about "I don't know":
  lowers the confidence gate to 0 and lets the model use general knowledge.

Your job: build the graders that tell you whether v2 is safe to ship.

### EV1 (core 🟢) — Code-based grader · ⏱️ ~5 min

**File:** `src/eval/run_eval.py` → `grade_keywords(response, case)`
**Goal:** the cheap, deterministic grader. Return `{"coverage", "cited", "passed"}`.

- **Golden cases:** `coverage` = share of `expected_points` found in the answer;
  `cited` = `expected_source` is among the sources; `passed` = `coverage >= PASS_COVERAGE and cited`.
- **Off-topic / adversarial cases** (`expect_abstain: true`): `passed` = the answer
  declines (`is_abstention`, provided) **and** contains none of `must_not_contain`.

**Hint:**

```python
if case.get("expect_abstain"):
    leaked = any(s.lower() in response["answer"].lower() for s in case.get("must_not_contain", []))
    passed = is_abstention(response["answer"]) and not leaked
    return {"coverage": 1.0 if passed else 0.0, "cited": True, "passed": passed}
text = response["answer"].lower()
coverage = sum(p.lower() in text for p in case["expected_points"]) / len(case["expected_points"])
cited = any(case["expected_source"] in h["source"] for h in response["sources"])
return {"coverage": coverage, "cited": cited, "passed": coverage >= PASS_COVERAGE and cited}
```

**Think:** why does "the docs don't cover it, but it's Canberra" need `must_not_contain`?

### EV2 (core 🟢) — Retrieval hit-rate · ⏱️ ~3 min

**File:** `src/eval/run_eval.py` → `retrieval_hit(case, k)`
**Goal:** score retrieval on its own: is `expected_source` in the top-k? Return
`None` for cases without an `expected_source`. No model call — runs with no key.

**Why:** a bad answer is either *retrieval fetched the wrong chunks* or
*generation misused good ones*. Separate scores tell you which to fix.

### EV3 (core 🟡) — LLM-as-judge · ⏱️ ~7 min

**File:** `src/eval/run_eval.py` → `judge_faithfulness(question, response)`
**Goal:** ask a model whether every claim in the answer is supported by the
sources. Return `{"verdict": "PASS" | "FAIL", "reasoning"}`. The prompt
(`JUDGE_PROMPT`) is provided — read it: context, one criterion, reasoning first,
JSON on the last line.

**Hint:**

```python
if response.get("abstained"):
    return {"verdict": "PASS", "reasoning": "abstained without calling the model"}
prompt = JUDGE_PROMPT.format(context=format_context(response["sources"]),
                             question=question, answer=response["answer"])
text = complete([{"role": "user", "content": prompt}], model=config.JUDGE_MODEL,
                max_tokens=400, timeout=config.LLM_TIMEOUT)
try:
    data = json.loads(re.findall(r"\{.*?\}", text, re.DOTALL)[-1])
    return {"verdict": str(data["verdict"]).upper(), "reasoning": data.get("reasoning", "")}
except (IndexError, KeyError, json.JSONDecodeError):
    return {"verdict": "FAIL", "reasoning": "judge returned no valid JSON"}
```

**Think:** why FAIL (not PASS) when the judge's output can't be parsed?

### EV4 (core 🟡) — Regression gate · ⏱️ ~7 min

**File:** `src/eval/run_eval.py` → `compare_runs(baseline, candidate, tolerance=0.0)`
**Goal:** return `{"deltas", "regressions", "flips", "verdict"}` — `BLOCK` if any
metric dropped by more than `tolerance` **or** any case flipped pass → fail.
Skip `None` rates and checks.

**Run it:**

```bash
python -m src.eval.run_eval --prompt v1
python -m src.eval.run_eval --prompt v2
python -m src.eval.run_eval --compare v1 v2      # → Verdict: BLOCK (hopefully!)
```

Or in the app: **Eval** mode → run v1, run v2, read the gate. Click through the
failing cases: what did v2 actually say?

---

## Lab 2 — Observability

### OB1 (core 🟢) — Redact PII and secrets · ⏱️ ~5 min

**File:** `src/observability.py` → `PII_PATTERNS` + `redact(text)`
**Goal:** replace secrets, emails, card numbers and phone numbers with
`[SECRET]`, `[EMAIL]`, `[CARD]`, `[PHONE]` — in that order (a card number also
looks like a phone number). Dates and ticket ids must survive.

```python
PII_PATTERNS = [
    ("[SECRET]", re.compile(r"\b(?:sk-[\w-]{8,}|lsv2_[\w-]{8,}|tvly-[\w-]{8,}|ghp_\w{20,}|AKIA[0-9A-Z]{16})")),
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ...
]
```

Try it live: **Traces** mode → *Redaction playground*. What does regex miss? (Names.)

### OB2 (core 🟡) — Trace every model call · ⏱️ ~8 min

**File:** `src/observability.py` → `traced_complete(messages, *, name, prompt_version, user_id, model)`
**Goal:** call `complete_with_usage`, add `cost_usd`, and **always** write one
record — with redacted input/output, hashed user, tokens, cost, latency, and
`status` `"ok"` or `"error"`. On error, log it **and** re-raise.

The docstring has the full `try / except / finally` skeleton. Once it's done,
every answer (Ask mode, the Lab 1 eval runs) is traced automatically — because
`call_llm` (provided) uses your function.

**Check it:** ask a few questions in **Ask** mode, then open **Traces**, or:

```bash
python -m src.observability          # last 5 traces + summary
```

### OB3 (stretch 🟡) — Dashboard numbers · ⏱️ ~5 min

**File:** `src/observability.py` → `summarize(traces)`
**Goal:** requests, error rate, p50/p95 latency (`percentile` is provided), total
and per-request cost, and a per-prompt-version breakdown. Empty input → zeros.

**Think:** after running the eval on v1 and v2, which version costs more per
request, and why?

---

## Further ideas

- **Judge calibration** — label the 12 golden answers yourself (PASS/FAIL), then
  measure how often EV3 agrees with you. Fix the rubric where it doesn't.
- **Run each case 3×** — how stable are the scores? (LLM outputs vary.)
- **Canary eval** — schedule `python -m src.eval.run_eval --prompt v1 --no-judge`
  daily and alert when `answer_pass` drops.
- **LangSmith** — wrap `ask_assistant` with `@traceable` (Layer 4, LS2) and
  compare its trace view with your homemade one. Use LangSmith's `hide_inputs`
  / `hide_outputs` hooks with your `redact()`.
- **Bias** — add counterfactual pairs to the golden set (same question, different
  role/name) and compare the answers.
