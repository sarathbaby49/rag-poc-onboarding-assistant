# 💰 Exercises — Model Selection, Customization & Cost (Layer 5)

Hands-on tasks for the Layer 5 session. The deck's story: our onboarding
assistant, rolled out org-wide, costs **~$8,000 a month** on one strong model.
Routing, trimming and caching bring that to **~$1,700**. Here you build the
pieces that make that cut, and watch your own bill move in the **Cost Lab**.

Exercises are a **menu**, not a checklist. **C1 is the anchor** (every other
exercise shows costs through it). Difficulty: 🟢 easy · 🟡 medium · 🔵 explore.

---

## Before you start

- [ ] `.env` has the gateway (`LITELLM_PROXY_API_BASE`, `LITELLM_PROXY_API_KEY`) and
      the three model tiers — copy them from `.env.example`:
      `CHEAP_MODEL`, `MID_MODEL`, `STRONG_MODEL` (names must match your proxy)
- [ ] `python -m checks.check_llm` passes
- [ ] Vector index built: `python -m src.ingest`

**Files you edit:** `src/cost.py` — all four exercises (C1, C2, C3, C4) live in
this one file, in that order. You don't edit `src/cost_helper.py` — it's the plumbing.

**Self-check** (instant ✅/❌, no key needed except the live caching check):

```bash
python -m checks.check_cost       # C1
python -m checks.check_routing    # C2 + C3
python -m checks.check_caching    # C4 (+ a live check if the gateway is set)
```

**Try it in the UI:**

```bash
streamlit run cost_lab.py
```

Stuck or out of time? Reference answers are in `solutions/cost.py` — try
first, then peek.

---

## ⏱️ Run of show (~40 min)

| Time | What |
|------|------|
| 0:00–0:05 | Open the Cost Lab → **Compare models**. Same question on three tiers: quality, speed, cost. No code. |
| 0:05–0:15 | **C1** — cost meter (everyone) · `check_cost` |
| 0:15–0:27 | **C2** — router · `check_routing` · try it in **Router** mode, watch “Saved” |
| 0:27–0:37 | Pick your depth: **C3** model per graph step · **C4** prompt caching |
| 0:37–0:40 | Recap: look at the projected monthly bill in the sidebar. What moved it most? |

Fast finishers: the **C2 bonus** (a classifier model instead of rules).

---

### C1 (core 🟢) — Cost meter · ⏱️ ~10 min

**Why:** you can't manage what you can't measure. Every other exercise reports
its savings through this function.

**File:** `src/cost.py` → `cost_of(usage, model)`

The price table (`PRICES`) and the lookup (`price_for`) are done. `usage` is a
normalized dict: fresh input tokens, output tokens, cache-read tokens and
cache-write tokens. Multiply each by the right price, add up, divide by
1,000,000. The docstring walks through it.

**Check:** `python -m checks.check_cost`. It includes the deck's own numbers:
15K in + 1K out on Opus 5.5 = $0.08, and the caching slide's 1 write + 9 reads = $0.043.

**In the lab:** every mode now shows dollars, and the sidebar projects your
session to 100,000 questions a month. In Compare mode, LiteLLM's own estimate
appears under yours as a cross-check.

---

### C2 (core 🟡) — Router · ⏱️ ~12 min

**Why:** routing is the biggest single cost lever. “What's the repo URL?” doesn't
need the strongest model.

**File:** `src/cost.py` → `pick_model(question)`

Right now it always returns `STRONG_MODEL`. Write a rule that sends short factual
lookups to `CHEAP_MODEL` and debugging, reasoning or code questions to
`STRONG_MODEL` (keywords, length, code-ish patterns — the docstring has ideas).

**Check:** `python -m checks.check_routing` runs 12 labelled questions; you need
10 right. It also prints what share went to the cheap tier.

**In the lab (Router mode):** ask a lookup and a debugging question. The lab
shows what the answer cost, what the same tokens would have cost on the strong
model, and the saving.

**Bonus 🔵:** replace the rules with a call to `CHEAP_MODEL` that labels the
question “easy” or “hard”. Is the extra call worth it? (Use C1 to find out.)

---

### C3 (medium 🟡) — A model per graph step · ⏱️ ~10 min

**Why:** model choice is per *step*, not per app. In the Layer 4 onboarding-plan
graph, 3 of 6 nodes need no model at all, and the ones that do need different tiers.

**File:** `src/cost.py` → `model_for_step(step, attempts)`

| Step | Task | Target |
|------|------|--------|
| `parse_request` | pull role + days out of “Create a 10-day onboarding plan for a new backend engineer.” | cheap |
| `draft_plan` | write the day-by-day plan | mid |
| `replan` | fix ungrounded steps / mentor feedback | mid; strong once `attempts >= 2` |

**Check:** `python -m checks.check_routing` (C3 section).

**In the lab (Plan graph mode):** run the request, approve or send feedback, and
read the per-step table: which model ran each step, its tokens and cost, next to
the cost if every step had used the strong model. The lab runs the reference
graph from `solutions/graph.py`, so this works even if your Layer 4 graph isn't finished.

---

### C4 (explore 🔵) — Prompt caching · ⏱️ ~8 min

**Why:** the repeated start of a prompt can cost ~10% of the normal price, but
only if it's byte-for-byte identical every time.

**File:** `src/cost.py` → `build_messages(question, hits)`

The current version works but breaks caching three ways: a timestamp at the
top, the question before the handbook, and nothing marked for caching. Rewrite
it: static system prompt + whole handbook first (one text block with
`cache_control`), dynamic context + question last. The docstring shows the exact shape.

**Check:** `python -m checks.check_caching` — structure checks run offline; a
live check sends two questions and looks for cache-read tokens.

**In the lab (Caching mode):** send two different questions. With C4 done, call 2
shows ~3,300 cache-read tokens, and the “without caching” column shows what you
saved. No hit? The checklist in the lab explains the usual causes.

---

## Further ideas (if the room is flying)

- **Batch it:** pre-generate onboarding plans for next week's joiners with a
  provider's batch API (50% off). What changes in the code, and what can't be batched?
- **Trim tool output:** the Layer 4 agent's `read_file` returns whole files.
  Return only the lines around a match, then compare tokens per question in LangSmith.
- **Fallbacks:** wrap a call with `llm.with_fallbacks([...])` and kill the
  primary model name to watch it fail over.

## Where this goes next

Every saving here assumed answer quality held. Sending half the questions to the
cheap tier is only smart if the cheap tier answers them well. **Layer 6
(evaluation)** is how you prove it: `src/eval/run_eval.py` and `golden_set.jsonl`.
