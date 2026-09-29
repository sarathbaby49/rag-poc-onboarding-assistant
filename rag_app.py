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

from src.rag_helper import (
    SessionMemory,  # window works even before M1 is done
    safe_answer,
    safe_answer_conversational,
    safe_answer_or_abstain,
    safe_answer_with_citations,
)

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

st.set_page_config(page_title="RAG Exercise Lab", page_icon="🧪")

# Align each radio's dot with the FIRST line of its label (not the vertical
# centre), so a label that wraps to two lines still lines up cleanly.
st.markdown(
    """
    <style>
      div[role="radiogroup"] > label { align-items: flex-start; }
      div[role="radiogroup"] > label > div:first-child { margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 RAG Exercise Lab")
st.caption("Test your src/rag.py answers (G1–G4). Unfinished exercises show a TODO instead of crashing.")

# label -> (mode key, one-line caption). Short labels stay on one line so the
# radio dot lines up with the text; the description rides along as a caption.
MODES = {
    "G1 · answer": ("answer", "grounded generation"),
    "G2 · answer_or_abstain": ("abstain", "honest “I don’t know”"),
    "G3 · used_sources": ("cited", "cite only what’s used"),
    "G4 · answer_conversational": ("conversational", "session memory"),
}

with st.sidebar:
    st.header("Which exercise?")
    labels = list(MODES.keys())
    mode_label = st.radio(
        "Function under test",
        labels,
        captions=[MODES[label][1] for label in labels],
    )
    mode = MODES[mode_label][0]
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
            elif mode == "cited":
                result = safe_answer_with_citations(prompt)
            else:  # conversational
                result = safe_answer_conversational(prompt, st.session_state.memory)

        st.markdown(result["answer"])

        if mode == "cited" and result.get("retrieved"):
            if result.get("g3_pending"):
                st.info(
                    "G3 (`used_sources`) isn't implemented yet — showing all "
                    f"{result['retrieved']} retrieved sources. Implement it to trim "
                    "these to only the ones the answer cited."
                )
            else:
                st.caption(
                    f"G3 filter: retrieved {result['retrieved']} → "
                    f"cited {len(result['sources'])}"
                )

        if result["sources"]:
            with st.expander("Sources"):
                for i, hit in enumerate(result["sources"], start=1):
                    st.markdown(f"**[{i}] {hit['source']}** — similarity {hit['score']:.2f}")
                    st.code(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
