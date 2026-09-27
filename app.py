"""Streamlit chat UI for the Onboarding Assistant demo.

Run:  streamlit run app.py

This is the face of the working RAG slice — type a question, get a cited answer.
Keep this open on screen during the session; it's your live demo.
"""

import streamlit as st
from dotenv import load_dotenv

from src.rag import answer

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

st.set_page_config(page_title="Onboarding Assistant", page_icon="🧭")
st.title("🧭 Engineering Onboarding Assistant")
st.caption("Ask about setup, architecture, or the codebase. Answers cite their sources.")

if "history" not in st.session_state:
    st.session_state.history = []

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
            result = answer(prompt)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for i, hit in enumerate(result["sources"], start=1):
                st.markdown(f"**[{i}] {hit['source']}** — similarity {hit['score']:.2f}")
                st.code(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
