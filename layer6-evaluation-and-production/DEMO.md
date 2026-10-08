# 🎬 Live demo script (presenter)

Three short live demos around the two labs, all in one app. Total demo time:
~12 minutes.

## Before the session (15 min, on VPN)

```bash
git checkout layer-6 && git pull
source .venv/bin/activate
python -m checks.check_llm                       # gateway OK?
python -m src.ingest                             # "Indexed 38 chunks"
python -m src.observability --clear              # empty trace log for a clean demo
rm -rf .eval_runs                                # no stale eval runs

# Optional: pre-run the v1 baseline so the room only waits for v2
python -m src.eval.run_eval --prompt v1 --solutions

streamlit run production_lab.py --server.fileWatcherType none
```

In the sidebar set **Code under test → Reference solutions**. Zoom the browser
to 125–150% so the room can read it.

> **If the gateway dies mid-session:** the Redaction playground and the
> retrieval-hit column work offline, saved eval runs in `.eval_runs/` still
> load, and the gate still compares them.

---

## Demo 1 — "Every answer has a trace" (after the warm-up, 3 min)

**Mode: Ask · prompt v1**

1. Type `What payment providers do we use?` → point at the caption under the
   answer: latency, tokens in/out, cost, sources.
2. Type `What is the capital of Australia?` → *"abstained: no model call · $0"*.
   The confidence gate said no before any tokens were spent.
3. Switch prompt to **v2**, ask the Australia question again → it happily
   answers from general knowledge. *"Is that a bug or a feature? Let's measure."*

## Demo 2 — "The judge and the gate" (Lab 1, ~6 min total)

**Mode: Eval** — before Lab 1, after the *LLM-as-judge* slide:

1. Select **v1**, judge on, press **Run eval on v1** (or show the pre-run one).
2. Walk the scoreboard: `answer_pass` (code grader — EV1), `retrieval_hit`
   (retrieval only — EV2), `faithful` (judge — EV3). Read one judge reasoning out loud.
3. *"Now you build these four functions."* → Lab 1.

At the end of Lab 1:

4. Select **v2**, press **Run eval on v2**. While it runs: *"Who'd ship v2? It's
   friendlier, never says I don't know."* Take a vote.
5. Scroll to **Regression gate** → Baseline v1, Candidate v2 → **BLOCK v2**.
   Read the flips: off-topic cases now answered from general knowledge, the judge
   failing faithfulness.
6. Punchline: *"Same users, same docs, one prompt edit — and the gate caught it
   before a client did."*

## Demo 3 — "The dashboard" (after Lab 2, 3 min)

**Mode: Traces**

1. The metrics row: requests, **p50 vs p95**, error rate, cost per request.
   *"p95 is what 1 in 20 users feels."*
2. *By prompt version*: compare v1 and v2 cost per request — v2 has no gate, so
   it pays for a model call on every off-topic question.
3. The table: inputs and outputs are already **redacted**, users are hashed ids.
4. **Redaction playground**: paste something with an email/phone/key — watch it
   disappear. Then type a name (`Priya from Acme`) — regex misses names. *"That's
   why teams use Presidio or a DLP service on top."*

---

## Fallback if the app won't load

The CLI tells the same story:

```bash
python -m src.eval.run_eval --prompt v1 --solutions
python -m src.eval.run_eval --prompt v2 --solutions
python -m src.eval.run_eval --compare v1 v2 --solutions
python -m src.observability --solutions
```
