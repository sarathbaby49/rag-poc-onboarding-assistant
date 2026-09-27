"""Retrieval Lab — a Streamlit playground for the ingestion → retrieval flow.

    streamlit run retrieval_lab.py          # no API key needed

What you can do here:
  - Adjust ingestion settings (chunk size, overlap, embedding model, chunker) and
    rebuild the index with one click.
  - Ask queries in a chat box and see the retrieved chunks with their score,
    cosine distance and source — the raw output of the vector search.
  - Visualise the whole vector store as a 2D map, with your last query projected
    onto it and the retrieved chunks ringed.

By default it talks to Chroma directly (like explore.py), so it works even before
the R1 exercise is done — great for demoing. Use the sidebar **Backend** selector to
switch to your own `semantic_search` (R1) / `hybrid_search` (R2) from src/retrieve.py
and watch your code power the chat and the vector map.
"""

from __future__ import annotations

import ast
import html
import inspect
import textwrap

import altair as alt
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src import config, ingest, retrieve

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
:root {
  --blue:#2563eb; --blue-dark:#1d4ed8; --blue-ink:#1e3a8a;
  --surface:#ffffff; --tint:#f2f6ff; --chunk:#e9f0fb;
  --border:#d7e0f0; --ink:#0f172a; --muted:#57678a;
}
.block-container {max-width: 900px;}
/* message rows */
.msg {display:flex; gap:12px; margin:16px 0; align-items:flex-start;}
.msg .avatar {flex:0 0 34px; width:34px; height:34px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:17px;}
.msg.usr {flex-direction:row-reverse;}
.msg.usr .avatar {background:#dbe6fb;}
.msg.bot .avatar {background:var(--blue); color:#fff;}
.bubble {border-radius:16px; padding:12px 16px; line-height:1.55; font-size:15px; color:var(--ink);}
.msg.usr .bubble {background:#dbe6fb; border-top-right-radius:5px; max-width:82%;}
.msg.bot .bubble {background:var(--surface); border:1px solid var(--border);
  border-top-left-radius:5px; width:100%; box-shadow:0 1px 3px rgba(15,23,42,0.06);}
.asst-head {font-size:11.5px; color:var(--muted); margin-bottom:12px; font-weight:700;
  text-transform:uppercase; letter-spacing:.5px;}
/* source cards */
.sources {display:flex; flex-direction:column; gap:10px;}
.src {border:1px solid var(--border); border-radius:12px; padding:10px 12px; background:var(--tint);}
.src-head {display:flex; align-items:center; gap:8px; flex-wrap:wrap;}
.rankbadge {background:var(--blue); color:#fff; border-radius:7px; padding:1px 8px;
  font-weight:700; font-size:12px;}
.srcpath {font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; color:var(--muted);}
.spacer {flex:1;}
.pill {background:#dbe6fb; color:var(--blue-ink); border-radius:999px; padding:2px 10px;
  font-weight:600; font-size:12px; white-space:nowrap;}
.pill.dist {background:#e7ecf5; color:var(--muted);}
pre.chunk {margin:9px 0 0; max-height:150px; overflow:auto; white-space:pre-wrap;
  word-break:break-word; background:var(--chunk); border:1px solid var(--border);
  border-radius:9px; padding:10px 12px; font-size:12px; line-height:1.5; color:var(--ink);}
/* rounded input + button */
div[data-testid="stForm"] {border:none; padding:0; background:transparent;}
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
  border-radius:999px !important; padding:13px 20px !important; font-size:15px !important;
  border:1px solid var(--border) !important; background:var(--surface) !important;}
div[data-testid="stFormSubmitButton"] button {border-radius:999px !important;
  padding:9px 24px !important; font-weight:600;}
/* exercise status badges (TODO tasks) */
.xrow {display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:0 0 10px;}
.xrow .lbl {font-size:12px; color:var(--muted);}
.xbadge {display:inline-block; font-size:11px; font-weight:700; padding:2px 9px;
  border-radius:999px; letter-spacing:.2px;}
.xbadge.todo {background:#fff4e5; color:#b45309; border:1px dashed #f59e0b;}
.xbadge.done {background:#dbe6fb; color:var(--blue-ink); border:1px solid var(--blue);}
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
    # retrieve.py caches its encoder + collection handle; refresh them so a
    # participant's semantic_search sees the newly-built index / model.
    for fn in (retrieve._encoder, retrieve._collection):
        try:
            fn.cache_clear()
        except Exception:  # noqa: BLE001
            pass
    ss.ingested_model, ss.chunk_size, ss.overlap, ss.smart_chunk = model, chunk_size, overlap, smart
    st.success(f"Re-ingested → {get_collection().count()} chunks.")
    st.rerun()


def _to_hits(raw: list[dict]) -> list[dict]:
    """Shape {text, source, score} results (from retrieve.py) for the UI."""
    return [
        {"rank": i + 1, "score": h["score"], "distance": 1 - h["score"],
         "source": h["source"], "text": h["text"]}
        for i, h in enumerate(raw)
    ]


def search(query: str, k: int, model_name: str, backend: str):
    # Always embed the query here so the Vector map can plot it, no matter which
    # backend produced the hits.
    qv = get_encoder(model_name).encode([query], normalize_embeddings=True)[0]

    if "semantic_search" in backend:
        hits = _to_hits(retrieve.semantic_search(query, k))       # participant's R1
    elif "hybrid_search" in backend:
        hits = _to_hits(retrieve.hybrid_search(query, k))         # participant's R2
    else:
        # Built-in: query Chroma directly — always works, even before R1 is done.
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


def exercise_status() -> tuple[bool, bool, bool]:
    """Detect (statically, from source) whether I1 / R1 / R2 are implemented yet."""
    def src(fn) -> str:
        try:
            return inspect.getsource(fn)
        except Exception:  # noqa: BLE001
            return ""

    def uses_bm25(fn) -> bool:
        # AST-based so we match a real BM25Okapi() *call*, not the word in the docstring.
        try:
            tree = ast.parse(textwrap.dedent(src(fn)))
        except Exception:  # noqa: BLE001
            return False
        return any(
            (isinstance(n, ast.Name) and n.id == "BM25Okapi")
            or (isinstance(n, ast.Attribute) and n.attr == "BM25Okapi")
            for n in ast.walk(tree)
        )

    i1 = "raise NotImplementedError" not in src(ingest.chunk_text_smart)
    r1 = "raise NotImplementedError" not in src(retrieve.semantic_search)
    r2 = uses_bm25(retrieve.hybrid_search)
    return i1, r1, r2


def chip(label: str, done: bool) -> str:
    cls = "done" if done else "todo"
    mark = "✓ done" if done else "🔧 TODO"
    return f'<span class="xbadge {cls}">{label} · {mark}</span>'


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
        f'<div class="asst-head">Top {len(turn["hits"])} chunks · via {html.escape(turn.get("backend", "built-in"))} · score = 1 − cosine distance</div>'
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
    docs, metas = data["documents"], data["metadatas"]
    ids = data.get("ids") or [""] * len(docs)
    retrieved = {h["text"] for h in hits} if hits else set()
    rank_by_text = {h["text"]: h["rank"] for h in hits} if hits else {}

    cols = {
        "x": xy[:, 0], "y": xy[:, 1],
        "id": ids,
        "source": [m.get("source", "") for m in metas],
        "chars": [len(d) for d in docs],
        "preview": [d.replace("\n", " ")[:90] for d in docs],
        "rank": [str(rank_by_text.get(d, "")) for d in docs],
        "retrieved": ["● retrieved" if d in retrieved else "" for d in docs],
    }
    tooltip = ["id", "source", "chars", "preview", "rank", "retrieved"]

    # Similarity of every chunk to the current query (both vectors are unit-length,
    # so a dot product is the cosine similarity = the score retrieval ranks by).
    if query_vec is not None:
        cols["sim to query"] = np.round(embs @ np.asarray(query_vec), 3)
        tooltip.insert(4, "sim to query")

    # Auto-include any extra metadata fields (e.g. a future `type`/`category`).
    for key in sorted({k for m in metas for k in m} - {"source"}):
        cols[key] = [str(m.get(key, "")) for m in metas]
        tooltip.append(key)

    # Short, full (un-truncated) file name for the legend; keep full path in tooltip.
    cols["file"] = [s.replace("data/sample_company/", "") for s in cols["source"]]
    df = pd.DataFrame(cols)

    # Click a legend entry to spotlight that file's chunks (others fade out).
    pick = alt.selection_point(fields=["file"], bind="legend")
    points = alt.Chart(df).mark_circle(size=150).encode(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("file", legend=alt.Legend(title="source file — click to locate",
                                                   labelLimit=1000, symbolLimit=100)),
        opacity=alt.condition(pick, alt.value(0.85), alt.value(0.07)),
        tooltip=tooltip,
    ).add_params(pick)
    layers = [points]
    retr = df[df["retrieved"] != ""]

    if query_vec is not None:
        q = pca.transform([query_vec])[0]

        # The "constellation": rays from the query to each retrieved chunk, a shared
        # blue halo on the matches, and their rank numbers — so the query and the
        # vectors it pulled read as one group and are easy to spot.
        if not retr.empty:
            rays = pd.DataFrame({
                "qx": q[0], "qy": q[1],
                "cx": retr["x"].to_numpy(), "cy": retr["y"].to_numpy(),
            })
            layers.append(
                alt.Chart(rays).mark_rule(color="#2563eb", strokeWidth=1.5,
                                          opacity=0.55, strokeDash=[4, 3])
                .encode(x="qx:Q", y="qy:Q", x2="cx:Q", y2="cy:Q")
            )
            layers.append(
                alt.Chart(retr).mark_point(shape="circle", size=340, filled=False,
                                           stroke="#2563eb", strokeWidth=3)
                .encode(x="x", y="y", tooltip=tooltip)
            )
            layers.append(
                alt.Chart(retr).mark_text(dy=-16, fontSize=13, fontWeight="bold",
                                          color="#1e3a8a").encode(x="x", y="y", text="rank")
            )

        qdf = pd.DataFrame({"x": [q[0]], "y": [q[1]], "label": [f"your query: {query_label}"]})
        layers.append(
            alt.Chart(qdf).mark_point(shape="diamond", size=520, color="#2563eb",
                                      filled=True, stroke="#1e3a8a", strokeWidth=2)
            .encode(x="x", y="y", tooltip=["label"])
        )

    st.altair_chart(alt.layer(*layers).interactive(), width="stretch")
    st.caption("Each dot is a chunk (colour = source file). Nearby dots mean similar "
               "meaning. The blue ◆ is your query; blue lines connect it to the chunks it "
               "retrieved (haloed, numbered by rank). **Click a file in the legend** to "
               "spotlight its chunks; hover any dot for its id, source, length, rank and "
               "similarity to your query.")


# --- session state (reflects the current on-disk index) ----------------------
ss = st.session_state
ss.setdefault("ingested_model", config.EMBED_MODEL)
ss.setdefault("chunk_size", config.CHUNK_SIZE)
ss.setdefault("overlap", config.CHUNK_OVERLAP)
ss.setdefault("smart_chunk", False)
ss.setdefault("history", [])  # [{"query", "hits", "query_vec"}]

# Live status of the exercise-backed controls (updates as participants implement them).
I1_DONE, R1_DONE, R2_DONE = exercise_status()


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
        "⚙ Boundary-aware chunking (exercise I1)", value=ss.smart_chunk,
        help="Uses your chunk_text_smart() from src/ingest.py instead of fixed-size slicing.",
    )
    st.markdown('<div class="xrow">' + chip("I1 chunk_text_smart", I1_DONE) + "</div>",
                unsafe_allow_html=True)

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
    # Backend picker + input on top …
    st.radio(
        "Retrieval backend",
        ["Built-in (always works)", "⚙ My semantic_search (R1)", "⚙ My hybrid_search (R2)"],
        key="backend", horizontal=True,
        help="Point the Lab at your own src/retrieve.py (R1/R2) to watch your code power "
             "the chat, or use the built-in direct-Chroma search that always works. "
             "The ⚙ options run code you implement in the exercises.",
    )
    st.markdown(
        '<div class="xrow"><span class="lbl">⚙ exercise-backed:</span>'
        + chip("R1 semantic_search", R1_DONE) + chip("R2 hybrid_search", R2_DONE)
        + "</div>",
        unsafe_allow_html=True,
    )
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
            try:
                with st.status("🔎 Fetching from the vector database…", expanded=True) as status:
                    st.write(f"Backend: **{ss.backend}** · embedding with `{ss.ingested_model}`…")
                    hits, qv = search(prompt, st.session_state.topk, ss.ingested_model, ss.backend)
                    st.write(f"Scored every chunk by cosine similarity → top {len(hits)}.")
                    status.update(label=f"✅ Retrieved {len(hits)} chunks from the vector DB",
                                  state="complete")
            except NotImplementedError:
                st.warning("That backend isn't implemented yet. Implement `semantic_search` (R1) "
                           "— and `hybrid_search` (R2) — in `src/retrieve.py`, or switch the "
                           "**Backend** to *Built-in* in the sidebar.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Retrieval failed: {type(e).__name__}: {e}")
            else:
                ss.history.insert(0, {"query": prompt, "hits": hits,
                                      "query_vec": qv.tolist(), "backend": ss.backend})
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
