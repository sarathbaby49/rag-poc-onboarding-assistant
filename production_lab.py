"""Production Lab — a Streamlit UI for the Layer 6 labs (and the live demo).

    streamlit run production_lab.py --server.fileWatcherType none

Modes (sidebar):
  - Ask      — chat with the assistant; every answer shows its trace (Lab 2: OB2)
  - Eval     — run the golden set on prompt v1 / v2, then SHIP or BLOCK (Lab 1: EV1–EV4)
  - Traces   — the dashboard: p50/p95, error rate, cost, redacted logs (Lab 2: OB1, OB3)

"Code under test" in the sidebar switches between your src/ exercises and the
reference solutions — presenters use it to demo the finished version. Unfinished
exercises show a friendly TODO instead of crashing, like the other labs.
"""

from __future__ import annotations

import importlib
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Production Lab", page_icon="🚦", layout="wide")


def _reload(module_name: str):
    """Force-reload a module so Streamlit picks up edits made during the exercises."""
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


@st.cache_data(ttl=30, show_spinner=False)
def _gateway_status() -> tuple[bool, str]:
    from src.layer6_support import gateway_reachable
    return gateway_reachable()


# ── Sidebar ──────────────────────────────────────────────────────────────────
MODES = {
    "Ask · traced assistant": ("ask", "chat; see latency, tokens and cost per answer"),
    "Eval · golden set": ("eval", "Lab 1: run v1 / v2, grade, then SHIP or BLOCK"),
    "Traces · dashboard": ("traces", "Lab 2: p50/p95, errors, cost, redacted logs"),
}

with st.sidebar:
    st.header("Mode")
    labels = list(MODES)
    mode_label = st.radio("Mode", labels, captions=[MODES[x][1] for x in labels], label_visibility="collapsed")
    mode = MODES[mode_label][0]

    st.divider()
    code_choice = st.radio("Code under test", ["My exercises (src/)", "Reference solutions"],
                           help="Presenters: switch to the reference solutions for the live demo.")
    use_solutions = code_choice == "Reference solutions"

# Reload in dependency order, then optionally swap in the solutions.
config = _reload("src.config")
_reload("src.llm")
obs = _reload("src.observability")
assistant = _reload("src.assistant")
run_eval = _reload("src.eval.run_eval")
support = _reload("src.layer6_support")
if use_solutions:
    for name in ("solutions.observability", "solutions.eval"):
        _reload(name)
    support.use_reference_solutions()

with st.sidebar:
    ok, msg = _gateway_status()
    (st.success if ok else st.error)(("LLM gateway: " + msg) if ok else msg, icon="🛰️")
    with st.expander("Exercise status", expanded=not use_solutions):
        for code, name, done in support.exercise_status():
            st.markdown(f"{'✅' if done else '⬜'} **{code}** `{name}`")
    st.caption(f"Model: `{config.LLM_MODEL}` · judge: `{config.JUDGE_MODEL}`")


def _todo(exc: NotImplementedError) -> None:
    st.warning(f"**Not built yet:** {exc}\n\nImplement it, or switch *Code under test* to the reference solutions.",
               icon="📝")


def _failed(exc: Exception) -> None:
    st.error(f"**{type(exc).__name__}:** {exc}\n\nIf this is a timeout, check the gateway (VPN?) with "
             "`python -m checks.check_llm`.", icon="🛑")


# ── Ask ──────────────────────────────────────────────────────────────────────
if mode == "ask":
    st.title("💬 Ask the assistant — with tracing")
    st.caption("Every answer goes through observability.call_llm, so once OB2 is done it lands in the trace log.")
    version = st.radio("Prompt version", list(assistant.PROMPTS), horizontal=True,
                       help="v1 = production. v2 = 'friendlier' prompt that answers from general knowledge.")

    st.session_state.setdefault("chat", [])
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                st.caption(turn["meta"])

    question = st.chat_input("Ask an onboarding question… try 'What payment providers do we use?' "
                             "or 'What is the capital of Australia?'")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and generating…"):
                    res = assistant.ask_assistant(question, prompt_version=version, user_id="lab-user")
                if res["abstained"]:
                    meta = "🙅 abstained: nothing cleared the confidence bar · no model call · $0"
                else:
                    u = res["usage"]
                    srcs = ", ".join(sorted({h["source"].split("/")[-1] for h in res["sources"]}))
                    meta = (f"🧾 prompt {version} · {u['latency_s']:.2f}s · {u['input_tokens']} in / "
                            f"{u['output_tokens']} out tokens · ${u['cost_usd']:.5f} · sources: {srcs}")
                st.markdown(res["answer"])
                st.caption(meta)
                st.session_state.chat.append({"role": "assistant", "content": res["answer"], "meta": meta})
            except Exception as exc:  # noqa: BLE001
                _failed(exc)
    if st.session_state.chat and st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ── Eval (Lab 1) ─────────────────────────────────────────────────────────────
elif mode == "eval":
    st.title("🧪 Eval — golden set & regression gate")
    cases = run_eval.load_golden()
    st.caption(f"{len(cases)} cases: "
               + ", ".join(f"{sum(c['type'] == t for c in cases)} {t}" for t in ("golden", "off_topic", "adversarial")))

    c1, c2, c3 = st.columns([1, 1, 2])
    version = c1.selectbox("Prompt version", list(assistant.PROMPTS))
    use_judge = c2.toggle("LLM judge (EV3)", value=True)
    if c3.button(f"▶️ Run eval on {version}", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        try:
            run = run_eval.run_suite(version, use_judge=use_judge,
                                     progress=lambda i, n, r: bar.progress(i / n, text=f"{i}/{n} · {r['id']}"))
            run_eval.save_run(run)
            st.session_state[f"run_{version}"] = run
            bar.empty()
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            _failed(exc)

    run = st.session_state.get(f"run_{version}") or run_eval.load_run(version)
    if run:
        st.subheader(f"Scoreboard · prompt {run['prompt_version']}")
        cols = st.columns(len(run["rates"]) + 1)
        for col, (metric, rate) in zip(cols, run["rates"].items()):
            col.metric(metric, "n/a" if rate is None else f"{rate:.0%}")
        cols[-1].metric("cost of this run", f"${run['total_cost_usd']:.4f}")
        mark = lambda v: "—" if v is None else ("✅" if v else "❌")  # noqa: E731
        st.dataframe(pd.DataFrame([{
            "id": r["id"], "type": r["type"], "question": r["question"],
            **{m: mark(v) for m, v in r["checks"].items()},
            "answer": r["answer"], "judge says": r["judge_reasoning"] or "",
        } for r in run["cases"]]), hide_index=True, use_container_width=True)
    else:
        st.info("No saved run for this version yet — press Run.")

    st.divider()
    st.subheader("🚦 Regression gate (EV4)")
    saved = [v for v in assistant.PROMPTS if run_eval.load_run(v)]
    if len(saved) < 2:
        st.caption("Run the eval on both v1 and v2 to compare them.")
    else:
        c1, c2 = st.columns(2)
        base_v = c1.selectbox("Baseline", saved, index=0)
        cand_v = c2.selectbox("Candidate", saved, index=len(saved) - 1)
        base, cand = run_eval.load_run(base_v), run_eval.load_run(cand_v)
        try:
            result = run_eval.compare_runs(base, cand)
            (st.error if result["verdict"] == "BLOCK" else st.success)(
                f"## {result['verdict']} {cand_v}", icon="🛑" if result["verdict"] == "BLOCK" else "🚀")
            cols = st.columns(len(result["deltas"]) or 1)
            for col, (metric, delta) in zip(cols, result["deltas"].items()):
                col.metric(metric, f"{cand['rates'][metric]:.0%}", f"{delta * 100:+.0f} pts")
            if result["flips"]:
                st.markdown("**Cases that went from pass to fail:**")
                by_id = {c["id"]: c for c in cand["cases"]}
                for flip in result["flips"]:
                    case = by_id[flip["id"]]
                    st.markdown(f"- `{flip['id']}` **{flip['metric']}** — *{case['question']}*  \n"
                                f"  → {case['answer'][:220]}")
        except NotImplementedError as exc:
            _todo(exc)

# ── Traces (Lab 2) ───────────────────────────────────────────────────────────
elif mode == "traces":
    st.title("🔭 Traces — the production dashboard")
    traces = obs.load_traces()
    c1, c2 = st.columns([4, 1])
    c1.caption(f"{len(traces)} traces in `.traces/traces.jsonl` — written by traced_complete (OB2).")
    if c2.button("🗑️ Clear traces"):
        obs.clear_traces()
        st.rerun()

    if not traces:
        st.info("No traces yet. Finish OB2, then ask a question or run an eval — and come back.")
    else:
        try:
            s = obs.summarize(traces)
            cols = st.columns(6)
            cols[0].metric("requests", s["requests"])
            cols[1].metric("p50 latency", f"{s['p50_latency_s']:.2f}s")
            cols[2].metric("p95 latency", f"{s['p95_latency_s']:.2f}s")
            cols[3].metric("error rate", f"{s['error_rate']:.0%}")
            cols[4].metric("total cost", f"${s['total_cost_usd']:.4f}")
            cols[5].metric("cost / request", f"${s['cost_per_request_usd']:.5f}")
            st.markdown("**By prompt version**")
            st.dataframe(pd.DataFrame([{"prompt_version": v, **d} for v, d in s["by_prompt_version"].items()]),
                         hide_index=True)
        except NotImplementedError as exc:
            _todo(exc)

        df = pd.DataFrame(traces)
        st.markdown("**Latency per request (s)** — the spikes are your p95")
        st.bar_chart(df["latency_s"])
        show = [c for c in ("timestamp", "name", "prompt_version", "status", "latency_s", "input_tokens",
                            "output_tokens", "cost_usd", "user", "input", "output", "error") if c in df]
        st.dataframe(df[show].iloc[::-1], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🧽 Redaction playground (OB1)")
    sample = st.text_area("Text that would be logged", "Hi, I'm priya.s@acme.com — call me on +44 7700 900123. "
                          "Card 4111 1111 1111 1111, key sk-live-abc123def456. Setup on 2026-10-07 for ACME-12.")
    try:
        st.code(obs.redact(sample), language=None)
    except NotImplementedError as exc:
        _todo(exc)
