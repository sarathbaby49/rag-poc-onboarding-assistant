"""Cost Lab — a Streamlit UI to test the Layer 5 exercises.

    streamlit run cost_lab.py

Pick a mode in the sidebar. Unfinished exercises show a hint instead of
crashing, like the other labs. Every call lands on the running bill in the
sidebar, projected to 100,000 questions a month (the deck's story).

Modes:
  - Compare models — one question, several models side by side (no code needed)
  - Router         — your pick_model (C2) chooses the model per question
  - Plan graph     — your model_for_step (C3) picks a model per graph step
  - Caching        — your build_messages (C4), sent twice, cache tokens compared
Costs come from your cost_of (C1).
"""

from __future__ import annotations

import importlib
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

from src import config

load_dotenv()

TIERS = {
    "cheap": config.CHEAP_MODEL,
    "mid": config.MID_MODEL,
    "strong": config.STRONG_MODEL,
}


def _reload(module_name: str):
    """Force-reload so edits to src/ show up without restarting Streamlit."""
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _money(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"${value:,.{digits}f}"


def _gateway_ready() -> bool:
    return bool(config.LITELLM_PROXY_API_BASE and config.LITELLM_PROXY_API_KEY)


st.set_page_config(page_title="Cost Lab", page_icon="💰", layout="wide")
st.title("💰 Cost Lab")
st.caption("Test your Layer 5 exercises — model choice, routing, caching, and what it all costs.")

ss = st.session_state
ss.setdefault("ledger", [])          # one row per question (or graph run)
ss.setdefault("graph_state", None)
ss.setdefault("graph_rows", [])
ss.setdefault("graph_msgs", [])
ss.setdefault("graph_thread", str(uuid.uuid4()))

MODES = {
    "Compare models": "one question, several models side by side",
    "Router · C2": "your pick_model chooses cheap or strong",
    "Plan graph · C3": "your model_for_step picks a model per step",
    "Caching · C4": "send your prompt twice and compare cache tokens",
}

# ── Sidebar: mode + running bill ─────────────────────────────────────────────
with st.sidebar:
    st.header("Which exercise?")
    mode = st.radio("Mode", list(MODES), captions=list(MODES.values()), label_visibility="collapsed")


if not _gateway_ready():
    st.warning(
        "The gateway isn't configured. Set `LITELLM_PROXY_API_BASE` and `LITELLM_PROXY_API_KEY` "
        "in `.env` (check with `python -m checks.check_llm`). The self-checks still run without it."
    )

cost_helper = _reload("src.cost_helper")
cost_mod = _reload("src.cost")


def _record(mode_name: str, label: str, result: dict) -> None:
    ss.ledger.append({"mode": mode_name, "label": label, "model": result["model"], "cost": result["cost"]})


def _usage_table(rows: list[dict]) -> None:
    st.dataframe(rows, hide_index=True, width="stretch")


# ── Mode: Compare models ─────────────────────────────────────────────────────
if mode == "Compare models":
    st.subheader("Same question, different models")
    st.write("No code needed. Watch quality, speed and cost move together (or not).")
    question = st.text_input("Question", "How do I set up the project locally?")
    picked = st.multiselect("Models", list(TIERS.values()), default=list(TIERS.values()),
                            format_func=cost_helper.short_name)
    if st.button("Run", type="primary", disabled=not (question and picked)):
        try:
            with st.spinner("Retrieving context…"):
                messages, hits = cost_helper.rag_messages(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval failed — did you run `python -m src.ingest`? ({exc})")
            st.stop()
        cols = st.columns(len(picked))
        for col, model in zip(cols, picked):
            with col:
                st.markdown(f"**{cost_helper.short_name(model)}**")
                try:
                    with st.spinner("Asking…"):
                        res = cost_helper.call(messages, model)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    continue
                _record("compare", question, res)
                u = res["usage"]
                st.metric("Cost", _money(res["cost"]), help=res["cost_note"] or None)
                st.caption(f"⏱️ {res['latency_s']:.1f}s · {u['input_tokens']:,} in · {u['output_tokens']:,} out")
                if res["litellm_cost"] is not None:
                    st.caption(f"LiteLLM's estimate: {_money(res['litellm_cost'])} (cross-check for C1)")
                st.markdown(res["text"])

# ── Mode: Router (C2) ─────────────────────────────────────────────────────────
elif mode == "Router · C2":
    st.subheader("Route each question to the cheapest model that can handle it")
    if cost_mod.pick_model("What's the repo URL?") == config.STRONG_MODEL:
        st.info("C2 isn't done yet: every question goes to the strong model. Edit `pick_model` in `src/cost.py`.")

    TIER_LABEL = {config.CHEAP_MODEL: "🟢 cheap", config.MID_MODEL: "🟡 mid", config.STRONG_MODEL: "🔴 strong"}

    def _tier_label(model: str) -> str:
        return TIER_LABEL.get(model, model)

    # Grounded in the actual ingested docs (data/sample_company), so retrieval
    # returns real hits, not just a plausible-sounding question.
    SAMPLES = [
        ("🔌 API port", "Which port does the API run on?"),               # setup.md
        ("💳 India payments", "Which payment provider do we use for customers in India?"),  # payments.md
        ("🔐 Refresh tokens", "Explain how refresh tokens work in our auth flow and the security trade-offs."),  # authentication.md
    ]

    st.caption("How `pick_model` routes a few samples right now:")
    st.dataframe(
        [{"question": q, "routed to": _tier_label(cost_mod.pick_model(q))} for _, q in SAMPLES],
        hide_index=True, width="stretch",
    )

    ss.setdefault("router_question", SAMPLES[0][1])
    st.caption("Try a sample, or paste your own below:")
    cols = st.columns(len(SAMPLES))
    for col, (label, sample_q) in zip(cols, SAMPLES):
        if col.button(label, width="stretch"):
            ss.router_question = sample_q

    question = st.text_input("Question", key="router_question",
                             help="Pick a sample above, or paste your own — a short lookup routes cheap, a debugging/reasoning question routes strong.")

    if st.button("Ask", type="primary", disabled=not question):
        chosen = cost_mod.pick_model(question)
        st.markdown(f"**Routed to:** {_tier_label(chosen)} — `{cost_helper.short_name(chosen)}`")
        try:
            with st.spinner("Retrieving context…"):
                messages, _ = cost_helper.rag_messages(question)
            with st.spinner(f"Asking {cost_helper.short_name(chosen)}…"):
                res = cost_helper.call(messages, chosen)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()
        _record("router", question, res)
        strong_cost, _ = cost_helper.safe_cost(res["usage"], config.STRONG_MODEL)
        c1, c2, c3 = st.columns(3)
        c1.metric("This answer cost", _money(res["cost"]), help=res["cost_note"] or None)
        c2.metric("Same tokens on the strong model", _money(strong_cost))
        if res["cost"] is not None and strong_cost:
            c3.metric("Saved", f"{1 - res['cost'] / strong_cost:.0%}")
        st.markdown(res["text"])

# ── Mode: Plan graph (C3) ─────────────────────────────────────────────────────
elif mode == "Plan graph · C3":
    st.subheader("A model per step in the onboarding-plan graph")
    st.write("plan (parse request → draft plan) → retrieve → check_grounding → replan? → mentor → publish. "
             "Only *parse_request*, *draft_plan* and *replan* call a model; your `model_for_step` picks which.")
    if cost_mod.model_for_step("parse_request") == config.STRONG_MODEL:
        st.info("C3 isn't done yet: every step uses the strong model. Edit `model_for_step` in `src/cost.py`.")

    if "graph_checkpointer" not in ss:
        from langgraph.checkpoint.memory import MemorySaver
        ss.graph_checkpointer = MemorySaver()

    def _turn(message: str) -> None:
        before = len(ss.graph_rows)
        try:
            with st.spinner("Running the graph…"):
                state, msgs, _ = cost_helper.routed_graph_turn(
                    message, ss.graph_state, ss.graph_thread, ss.graph_checkpointer, ss.graph_rows
                )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        ss.graph_state = state
        ss.graph_msgs += msgs
        new_rows = ss.graph_rows[before:]
        costs = [r["cost"] for r in new_rows]
        ss.ledger.append({"mode": "graph", "label": message, "model": "per step",
                          "cost": None if any(c is None for c in costs) else sum(costs)})

    waiting = bool(ss.graph_state and ss.graph_state.get("_waiting_for_mentor"))
    if ss.graph_state is None:
        request = st.text_input("Request", "Create a 10-day onboarding plan for a new backend engineer.")
        if st.button("Run", type="primary", disabled=not request):
            _turn(request)
            st.rerun()
    elif waiting:
        a, b = st.columns([1, 3])
        if a.button("✅ Approve", type="primary", width="stretch"):
            _turn("approve")
            st.rerun()
        feedback = b.text_input("…or mentor feedback (triggers a replan)", placeholder="Add pairing on Day 3")
        if feedback and b.button("📝 Send feedback"):
            _turn(feedback)
            st.rerun()
    if ss.graph_state is not None and st.button("🔄 New plan"):
        ss.graph_state, ss.graph_rows, ss.graph_msgs = None, [], []
        ss.graph_thread = str(uuid.uuid4())
        ss.pop("graph_checkpointer", None)
        st.rerun()

    if ss.graph_rows:
        st.markdown("**What each step cost**")
        rows = [
            {"step": r["step"], "replan attempt": r["attempt"] if r["step"] == "replan" else "",
             "model": cost_helper.short_name(r["model"]), "input": r["input_tokens"],
             "output": r["output_tokens"], "cost": _money(r["cost"])}
            for r in ss.graph_rows
        ]
        _usage_table(rows)
        total = [r["cost"] for r in ss.graph_rows]
        if all(c is not None for c in total):
            all_strong = sum(cost_helper.safe_cost(r, config.STRONG_MODEL)[0] or 0 for r in ss.graph_rows)
            c1, c2 = st.columns(2)
            c1.metric("This run", _money(sum(total)))
            c2.metric("Same tokens, strong model everywhere", _money(all_strong))
    for msg in ss.graph_msgs:
        with st.chat_message("assistant"):
            st.markdown(msg)

# ── Mode: Caching (C4) ────────────────────────────────────────────────────────
elif mode == "Caching · C4":
    st.subheader("Static first, dynamic last")
    prefix_tokens = len(cost_mod.SYSTEM_PROMPT + cost_mod.handbook()) // 4
    st.write(f"Your prompt ships the whole team handbook (~{prefix_tokens:,} tokens) on every call. "
             "With caching, the second call should read it from the cache at ~10% of the price.")
    model = st.selectbox("Model", list(TIERS.values()), index=1, format_func=cost_helper.short_name)
    q1 = st.text_input("First question", "What's the repo URL?")
    q2 = st.text_input("Second question (different on purpose)", "Which port does the API run on?")
    if st.button("Send both", type="primary", disabled=not (q1 and q2)):
        results = []
        for q in (q1, q2):
            try:
                with st.spinner(f"Sending: {q}"):
                    results.append(cost_helper.call(cost_mod.build_messages(q), model, max_tokens=200))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.stop()
        for q, r in zip((q1, q2), results):
            _record("caching", q, r)
        rows = []
        for i, r in enumerate(results, 1):
            u = r["usage"]
            no_cache = {"input_tokens": u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"],
                        "output_tokens": u["output_tokens"]}
            rows.append({"call": i, "fresh input": u["input_tokens"], "cache write": u["cache_write_tokens"],
                         "cache read": u["cache_read_tokens"], "output": u["output_tokens"],
                         "cost": _money(r["cost"]),
                         "without caching": _money(cost_helper.safe_cost(no_cache, model)[0])})
        _usage_table(rows)
        if results[1]["usage"]["cache_read_tokens"] > 0:
            st.success("Call 2 read the handbook from the cache. 🎉")
        else:
            st.warning(
                "No cache hit on call 2. Is C4 done (`python -m checks.check_caching`)? "
                "Both calls must start with byte-identical text marked with cache_control, "
                "within the cache lifetime (5 minutes)."
            )
        for q, r in zip((q1, q2), results):
            with st.expander(q):
                st.markdown(r["text"])

# ── Sidebar bill (rendered last so it includes this run's calls) ─────────────
with st.sidebar:
    st.divider()
    st.header("🧾 Session bill")
    priced = [r["cost"] for r in ss.ledger if r["cost"] is not None]
    total = sum(priced)
    st.metric("Questions", len(ss.ledger))
    st.metric("Spent this session", _money(total))
    if priced:
        avg = total / len(priced)
        per_month = st.number_input("Questions per month", 1_000, 10_000_000, 100_000, step=10_000)
        st.metric("Projected monthly bill", _money(avg * per_month, 0), help="Average cost per question × monthly volume")
    elif ss.ledger:
        st.caption("Finish C1 (`src/cost.py`) to see dollar amounts.")
    if st.button("🧹 Reset bill", width="stretch"):
        ss.ledger = []
        st.rerun()
