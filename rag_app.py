"""RAG exercise playground — a Streamlit UI to test src/rag.py (G1–G4).

    streamlit run rag_app.py

Like retrieval_lab.py is for the R exercises, this is for the G (generation)
exercises. Pick which function to exercise in the sidebar, ask a question, and
see the answer + the sources it cited.

It goes through src/rag_helper.py, which catches NotImplementedError from any
unfinished link in the chain (R1/R3/M1/G1–G4). So the page never crashes: an
unfinished exercise shows a friendly "implement this next" message instead, and
the UI turns into a live progress board as you complete each one.

Note: G1/G2/G4 call the LLM, so they need ANTHROPIC_API_KEY in .env. The G2
abstain path (off-topic question) short-circuits WITHOUT a model call.
"""

import streamlit as st
from dotenv import load_dotenv

from src.memory import SessionMemory
from src.rag_helper import (
    safe_answer,
    safe_answer_conversational,
    safe_answer_or_abstain,
)

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

st.set_page_config(page_title="RAG Exercise Lab", page_icon="🧪")
st.title("🧪 RAG Exercise Lab")
st.caption("Test your src/rag.py answers (G1–G4). Unfinished exercises show a TODO instead of crashing.")

MODES = {
    "G1 — answer (grounded generation)": "answer",
    "G2 — answer_or_abstain (honest I-don't-know)": "abstain",
    "G4 — answer_conversational (session memory)": "conversational",
}

with st.sidebar:
    st.header("Which exercise?")
    mode_label = st.radio("Function under test", list(MODES.keys()))
    mode = MODES[mode_label]
    st.divider()
    if st.button("🧹 Clear conversation"):
        st.session_state.history = []
        st.session_state.memory = SessionMemory()
        st.rerun()
    st.caption(
        "G4 threads the last few turns via SessionMemory (M1). "
        "Ask a follow-up like *'what about the tests?'* to see memory at work."
    )

if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()

# Replay the conversation so far.
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input("e.g. How do I set up my local environment?"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            if mode == "answer":
                result = safe_answer(prompt)
            elif mode == "abstain":
                result = safe_answer_or_abstain(prompt)
            else:  # conversational
                result = safe_answer_conversational(prompt, st.session_state.memory)

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for i, hit in enumerate(result["sources"], start=1):
                    st.markdown(f"**[{i}] {hit['source']}** — similarity {hit['score']:.2f}")
                    st.code(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
