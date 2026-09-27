"""Retrieval Lab — a Streamlit playground for the ingestion → retrieval flow.

    streamlit run retrieval_lab.py          # no API key needed

What you can do here:
  - Adjust ingestion settings (chunk size, overlap, embedding model, chunker) and
    rebuild the index with one click.
  - Ask queries in a chat box and see the retrieved chunks with their score,
    cosine distance and source — the raw output of the vector search.
  - Visualise the whole vector store as a 2D map, with your last query projected
    onto it and the retrieved chunks ringed.

It talks to Chroma directly (like explore.py), so it works even before the R1
`semantic_search` exercise is done — great for demoing and playing around.
"""

from __future__ import annotations

import html

import altair as alt
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src import config, ingest

# Small, fast, 384-dim models — all download in seconds on first use.
MODEL_CHOICES = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
    "multi-qa-MiniLM-L6-cos-v1",
]

st.set_page_config(page_title="Retrieval Lab", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 900px;}
/* message rows */
.msg {display:flex; gap:12px; margin:16px 0; align-items:flex-start;}
.msg .avatar {flex:0 0 34px; width:34px; height:34px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:17px;}
.msg.usr {flex-direction:row-reverse;}
.msg.usr .avatar {background:rgba(130,130,150,0.20);}
.msg.bot .avatar {background:#05954e;}
.bubble {border-radius:16px; padding:11px 15px; line-height:1.55; font-size:15px;}
.msg.usr .bubble {background:rgba(130,130,150,0.15); border-top-right-radius:5px; max-width:82%;}
.msg.bot .bubble {background:rgba(130,130,150,0.08); border:1px solid rgba(130,130,150,0.20);
  border-top-left-radius:5px; width:100%;}
.asst-head {font-size:11.5px; opacity:.6; margin-bottom:12px; font-weight:700;
  text-transform:uppercase; letter-spacing:.5px;}
/* source cards */
.sources {display:flex; flex-direction:column; gap:10px;}
.src {border:1px solid rgba(130,130,150,0.22); border-radius:12px; padding:10px 12px;}
.src-head {display:flex; align-items:center; gap:8px; flex-wrap:wrap;}
.rankbadge {background:#05954e; color:#fff; border-radius:7px; padding:1px 8px;
  font-weight:700; font-size:12px;}
.srcpath {font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; opacity:.85;}
.spacer {flex:1;}
.pill {background:#e6f6ee; color:#036334; border-radius:999px; padding:2px 10px;
  font-weight:600; font-size:12px; white-space:nowrap;}
.pill.dist {background:rgba(130,130,150,0.18); color:inherit;}
pre.chunk {margin:9px 0 0; max-height:150px; overflow:auto; white-space:pre-wrap;
  word-break:break-word; background:rgba(130,130,150,0.10); border-radius:9px;
  padding:10px 12px; font-size:12px; line-height:1.5;}
/* rounded input + button */
div[data-testid="stForm"] {border:none; padding:0; background:transparent;}
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
  border-radius:999px !important; padding:13px 20px !important; font-size:15px !important;
  border:1px solid rgba(130,130,150,0.35) !important;}
div[data-testid="stFormSubmitButton"] button {border-radius:999px !important;
  padding:9px 24px !important; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)


# --- resources ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def index_exists() -> bool:
    try:
        get_collection().count()
        return True
    except Exception:
        return False


def reingest(chunk_size: int, overlap: int, model: str, smart: bool) -> None:
    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = overlap
    config.EMBED_MODEL = model
    chunker = ingest.chunk_text_smart if smart else ingest.chunk_text
    try:
        with st.spinner(f"Re-ingesting with {model} (size={chunk_size}, overlap={overlap})…"):
            ingest.build_index(chunker=chunker)
    except NotImplementedError:
        st.error("Boundary-aware chunking (exercise I1) isn't implemented yet — "
                 "implement `chunk_text_smart` in `src/ingest.py`, or uncheck it.")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Re-ingest failed: {type(e).__name__}: {e}")
        return
    ss.ingested_model, ss.chunk_size, ss.overlap, ss.smart_chunk = model, chunk_size, overlap, smart
    st.success(f"Re-ingested → {get_collection().count()} chunks.")
    st.rerun()


def search(query: str, k: int, model_name: str):
    qv = get_encoder(model_name).encode([query], normalize_embeddings=True)[0]
    res = get_collection().query(
        query_embeddings=[qv.tolist()], n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {"rank": i + 1, "score": 1 - d, "distance": d, "source": m["source"], "text": t}
        for i, (t, m, d) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        )
    ]
    return hits, qv


def turn_html(turn: dict) -> str:
    """Render one query + its retrieved chunks as a Claude/Gemini-style exchange."""
    cards = []
    for h in turn["hits"]:
        cards.append(
            '<div class="src"><div class="src-head">'
            f'<span class="rankbadge">#{h["rank"]}</span>'
            f'<span class="srcpath">{html.escape(h["source"])}</span>'
            '<span class="spacer"></span>'
            f'<span class="pill">score {h["score"]:.2f}</span>'
            f'<span class="pill dist">dist {h["distance"]:.2f}</span>'
            f'</div><pre class="chunk">{html.escape(h["text"])}</pre></div>'
        )
    return (
        '<div class="msg usr"><div class="avatar">🧑</div>'
        f'<div class="bubble">{html.escape(turn["query"])}</div></div>'
        '<div class="msg bot"><div class="avatar">🔎</div><div class="bubble">'
        f'<div class="asst-head">Top {len(turn["hits"])} chunks · score = 1 − cosine distance</div>'
        f'<div class="sources">{"".join(cards)}</div>'
        '</div></div>'
    )


def render_map(query_vec=None, query_label: str | None = None, hits: list[dict] | None = None) -> None:
    data = get_collection().get(include=["embeddings", "documents", "metadatas"])
    embs = np.array(data["embeddings"])
    if len(embs) < 2:
        st.info("Not enough chunks to draw a map.")
        return
    pca = PCA(n_components=2).fit(embs)
    xy = pca.transform(embs)
    retrieved = {h["text"] for h in hits} if hits else set()
    df = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "source": [m["source"] for m in data["metadatas"]],
        "preview": [d.replace("\n", " ")[:90] for d in data["documents"]],
        "retrieved": ["● retrieved" if d in retrieved else "" for d in data["documents"]],
    })

    points = alt.Chart(df).mark_circle(size=140, opacity=0.75).encode(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("source", legend=alt.Legend(title="source file")),
        tooltip=["source", "preview", "retrieved"],
    )
    layers = [points]

    if retrieved:
        rings = alt.Chart(df[df["retrieved"] != ""]).mark_point(
            size=340, stroke="#101928", strokeWidth=2, filled=False
        ).encode(x="x", y="y")
        layers.append(rings)

    if query_vec is not None:
        q = pca.transform([query_vec])[0]
        qdf = pd.DataFrame({"x": [q[0]], "y": [q[1]], "label": [f"your query: {query_label}"]})
        qmark = alt.Chart(qdf).mark_point(
            shape="diamond", size=500, color="#05954e", filled=True
        ).encode(x="x", y="y", tooltip=["label"])
        layers.append(qmark)

    st.altair_chart(alt.layer(*layers).interactive(), width="stretch")
    st.caption("Each dot is a chunk (colour = source file). Nearby dots mean similar "
               "meaning. The green ◆ is your last query; ringed dots are the chunks it "
               "retrieved — they're the ones sitting closest to it.")


# --- session state (reflects the current on-disk index) ----------------------
ss = st.session_state
ss.setdefault("ingested_model", config.EMBED_MODEL)
ss.setdefault("chunk_size", config.CHUNK_SIZE)
ss.setdefault("overlap", config.CHUNK_OVERLAP)
ss.setdefault("smart_chunk", False)
ss.setdefault("history", [])  # [{"query", "hits", "query_vec"}]


# --- sidebar: ingestion controls ---------------------------------------------
with st.sidebar:
    st.header("⚙️ Ingestion settings")
    model = st.selectbox(
        "Embedding model", MODEL_CHOICES,
        index=MODEL_CHOICES.index(ss.ingested_model) if ss.ingested_model in MODEL_CHOICES else 0,
    )
    chunk_size = st.slider("Chunk size (chars)", 100, 2000, int(ss.chunk_size), 50)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, int(ss.overlap), 10)
    smart = st.checkbox(
        "Boundary-aware chunking (exercise I1)", value=ss.smart_chunk,
        help="Uses your chunk_text_smart() from src/ingest.py instead of fixed-size slicing.",
    )

    pending = (model, chunk_size, overlap, smart) != (
        ss.ingested_model, ss.chunk_size, ss.overlap, ss.smart_chunk)
    if pending:
        st.caption("⚠️ Settings changed — re-ingest to apply them.")

    if st.button("🔁 Re-ingest data", type="primary", width="stretch"):
        reingest(chunk_size, overlap, model, smart)

    st.divider()
    st.subheader("🔎 Retrieval")
    st.slider("Top-k (chunks to retrieve)", 1, 10, int(config.TOP_K), key="topk")

    st.divider()
    st.subheader("📦 Current index")
    if index_exists():
        col = get_collection()
        n = col.count()
        dim = len(col.get(include=["embeddings"], limit=1)["embeddings"][0]) if n else 0
        st.metric("Chunks stored", n)
        st.write(f"**Model:** `{ss.ingested_model}`")
        st.write(f"**Vector dim:** {dim}")
        st.write(f"**Chunk size / overlap:** {ss.chunk_size} / {ss.overlap}")
        st.write(f"**Chunker:** {'boundary-aware (I1)' if ss.smart_chunk else 'fixed-size'}")
    else:
        st.warning("No index yet — click **Re-ingest data**.")


# --- main --------------------------------------------------------------------
st.title("🔎 Retrieval Lab")
st.caption("Play with the ingestion → retrieval flow — local embeddings + vector search, no API key.")

tab_chat, tab_map = st.tabs(["🔎 Retrieve", "🗺️ Vector map"])

with tab_chat:
    # Input on top …
    with st.form("ask", clear_on_submit=True):
        prompt = st.text_input(
            "query", placeholder="Ask the knowledge base…  e.g. how does checkout work?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔎  Search", type="primary")

    if submitted and prompt:
        if not index_exists():
            st.error("No index yet — set your parameters in the sidebar and click **Re-ingest data**.")
        else:
            with st.status("🔎 Fetching from the vector database…", expanded=True) as status:
                st.write(f"Embedding your query with `{ss.ingested_model}`…")
                hits, qv = search(prompt, st.session_state.topk, ss.ingested_model)
                st.write(f"Scored every chunk by cosine similarity → top {len(hits)}.")
                status.update(label=f"✅ Retrieved {len(hits)} chunks from the vector DB",
                              state="complete")
            ss.history.insert(0, {"query": prompt, "hits": hits, "query_vec": qv.tolist()})
            st.rerun()

    if ss.history and st.button("🧹 Clear chat"):
        ss.history = []
        st.rerun()

    # … conversation below, newest first (latest exchange sits right under the input)
    st.write("")
    if not ss.history:
        st.markdown(
            '<div style="opacity:.55; text-align:center; padding:32px 0; font-size:15px;">'
            "Ask a question to search the knowledge base — you'll see the top matching "
            "chunks with their similarity scores.</div>",
            unsafe_allow_html=True,
        )
    for turn in ss.history:
        st.markdown(turn_html(turn), unsafe_allow_html=True)

with tab_map:
    if not index_exists():
        st.info("Re-ingest to build the vector store, then come back to see the map.")
    elif ss.history:
        last = ss.history[0]  # newest-first
        render_map(np.array(last["query_vec"]), last["query"], last["hits"])
    else:
        render_map()
        st.caption("Ask a query on the Retrieve tab to see it projected onto the map.")
