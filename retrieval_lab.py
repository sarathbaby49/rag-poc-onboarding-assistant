"""Retrieval Lab — a Streamlit playground for the ingestion → retrieval flow.

    streamlit run retrieval_lab.py          # no API key needed

What you can do here:
  - Adjust ingestion settings (chunk size, overlap, embedding model, chunker) and
    rebuild the index with one click.
  - Ask queries in a chat box and see the retrieved chunks with their score,
    cosine distance and source — the raw output of the vector search.
  - Visualise the whole vector store as a 2D map, with your last query projected
    onto it and the retrieved chunks ringed.
  - Chat with the full **Assistant** — the actual RAG chatbot: retrieval feeds a
    grounded, cited answer from the model, with session + long-term memory wired in
    (this tab needs the LiteLLM gateway in .env; the others run key-free).

By default it talks to Chroma directly (like explore.py), so it works even before
the R1 exercise is done — great for demoing. Use the sidebar **Backend** selector to
switch to your own `semantic_search` (R1) / `hybrid_search` (R2) from src/retrieve.py
and watch your code power the chat and the vector map.
"""

from __future__ import annotations

import ast
import html
import inspect
import json
import textwrap

import altair as alt
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src import config, ingest, llm, memory, rag, retrieve

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
    return SentenceTransformer(model_name, device=config.EMBED_DEVICE)


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def index_exists() -> bool:
    try:
        get_collection().count()
        return True
    except Exception:
        return False


def reingest(chunk_size: int, overlap: int, model: str, fixed: bool) -> None:
    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = overlap
    config.EMBED_MODEL = model
    # Default is boundary-aware; the checkbox lets you fall back to fixed-size to compare.
    chunker = ingest.chunk_text if fixed else ingest.chunk_text_smart
    try:
        with st.spinner(f"Re-ingesting with {model} (size={chunk_size}, overlap={overlap})…"):
            ingest.build_index(chunker=chunker)
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
    ss.ingested_model, ss.chunk_size, ss.overlap, ss.fixed_chunk = model, chunk_size, overlap, fixed
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


def get_retriever(name: str):
    """Map the Assistant's retrieval picker to a function rag.answer can call."""
    return retrieve.hybrid_search if name == "hybrid_search" else retrieve.semantic_search


GEN_GATEWAY = "Gateway (LiteLLM)"  # display label for how answers are generated


def who(name: str) -> str:
    """Normalise a joinee name into a stable key for their on-disk files."""
    return (name or "guest").strip() or "guest"


def mem_path(name: str):
    """Where this joinee's long-term MemoryStore lives (next to their profile)."""
    return memory.PROFILE_DIR / f"{who(name)}.memories.json"


def load_or_create_profile(name: str, role: str) -> memory.JoineeProfile:
    """Long-term profile for this joinee, loaded fresh from disk on every query.

    Seen before -> their saved role + completed steps come back (persistence in
    action). New name -> we create and save a profile so it's there next time.
    """
    key = who(name)
    try:
        return memory.JoineeProfile.load(key)
    except FileNotFoundError:
        p = memory.JoineeProfile(name=key, role=(role or "engineer").strip() or "engineer")
        p.save()
        return p


def profile_lines(p: memory.JoineeProfile) -> list[str]:
    """Format a profile as short facts for the prompt's system context."""
    done = ", ".join(p.completed_steps) if p.completed_steps else "nothing yet"
    return [f"Name: {p.name}", f"Role: {p.role}", f"Onboarding completed: {done}"]


# --- persistent search history (survives restarts, unlike session state) ------
SEARCH_HISTORY_PATH = config.BASE_DIR / ".search_history.json"
MAX_SEARCH_HISTORY = 50


def load_search_history() -> list[str]:
    """Read past searches from disk; [] if the file is missing or unreadable."""
    try:
        data = json.loads(SEARCH_HISTORY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return []
    return [str(q) for q in data][:MAX_SEARCH_HISTORY] if isinstance(data, list) else []


def save_search_history(items: list[str]) -> None:
    """Persist the search history to disk (best-effort)."""
    try:
        SEARCH_HISTORY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")
    except OSError:
        pass


def record_search(query: str) -> None:
    """Add a query to the front of the history (most-recent-first, de-duplicated)."""
    q = query.strip()
    if not q:
        return
    hist = [h for h in ss.search_history if h != q]  # drop an earlier copy
    hist.insert(0, q)
    ss.search_history = hist[:MAX_SEARCH_HISTORY]
    save_search_history(ss.search_history)


def run_search(query: str) -> None:
    """Run a knowledge-base search and record it in the persistent history."""
    query = (query or "").strip()
    if not query:
        return
    if not index_exists():
        st.error("No index yet — set your parameters in the sidebar and click **Re-ingest data**.")
        return
    try:
        with st.status("🔎 Fetching from the vector database…", expanded=True) as status:
            st.write(f"Backend: **{ss.backend}** · embedding with `{ss.ingested_model}`…")
            hits, qv = search(query, st.session_state.topk, ss.ingested_model, ss.backend)
            st.write(f"Scored every chunk by cosine similarity → top {len(hits)}.")
            status.update(label=f"✅ Retrieved {len(hits)} chunks from the vector DB",
                          state="complete")
    except NotImplementedError:
        st.warning("That backend isn't implemented yet. Implement `semantic_search` (R1) "
                   "— and `hybrid_search` (R2) — in `src/retrieve.py`, or switch the "
                   "**Backend** to *Built-in* in the sidebar.")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Retrieval failed: {type(e).__name__}: {e}")
        return
    ss.history.insert(0, {"query": query, "hits": hits,
                          "query_vec": qv.tolist(), "backend": ss.backend})
    record_search(query)  # persist to disk so it's here next launch
    st.rerun()


def exercise_status() -> tuple[bool, bool]:
    """Detect (statically, from source) whether R1 / R2 are implemented yet."""
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

    r1 = "raise NotImplementedError" not in src(retrieve.semantic_search)
    r2 = uses_bm25(retrieve.hybrid_search)
    return r1, r2


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


ROLE_STYLE = {
    "system": ("🛡️", "#fdf2e9", "#b45309", "System"),
    "user": ("🧑", "#dbe6fb", "#1e3a8a", "User"),
    "assistant": ("🤖", "#eef7ee", "#166534", "Assistant"),
}


def render_context_window(messages: list[dict]) -> None:
    """Show the exact payload (list of role/content messages) sent to the LLM.

    This is the context window: what the model actually saw. We render each
    message with its role, a rough token estimate (~chars/4) and the raw content,
    so participants can see how the system prompt, recalled memories, session
    history and the retrieved context are stitched together into one request.
    """
    if not messages:
        st.caption("No payload captured for this turn.")
        return

    total_chars = sum(len(m.get("content", "")) for m in messages)
    st.caption(
        f"**{len(messages)} messages** · ~{total_chars:,} chars · "
        f"~{total_chars // 4:,} tokens (est.) — this is the entire context window "
        "sent to the model in one request."
    )
    for i, m in enumerate(messages, start=1):
        role = m.get("role", "?")
        content = m.get("content", "")
        icon, bg, ink, label = ROLE_STYLE.get(role, ("•", "#eef1f6", "#57678a", role.title()))
        chars = len(content)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:12px 0 4px;">'
            f'<span style="background:{bg};color:{ink};border-radius:7px;padding:2px 10px;'
            f'font-weight:700;font-size:12px;">{icon} {i}. {label}</span>'
            f'<span style="color:#57678a;font-size:12px;">{chars:,} chars · ~{chars // 4:,} tokens</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.code(content, language="markdown")


def render_map(query_vec=None, query_label: str | None = None, hits: list[dict] | None = None,
               memories: list[dict] | None = None) -> None:
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

    # Long-term memories (MemoryStore) — same embedding space as the chunks, so we
    # project them through the SAME PCA (exactly like the query ◆) and draw them as a
    # distinct amber ★ layer. Only vectors whose dimensionality matches the current
    # index are plotted (a memory embedded with a different model can't share this space).
    if memories:
        dim = embs.shape[1]
        mem_ok = [m for m in memories
                  if isinstance(m.get("embedding"), list) and len(m["embedding"]) == dim]
        if mem_ok:
            mxy = pca.transform(np.array([m["embedding"] for m in mem_ok]))
            mdf = pd.DataFrame({
                "x": mxy[:, 0], "y": mxy[:, 1],
                "memory": [m["text"].replace("\n", " ")[:90] for m in mem_ok],
            })
            layers.append(
                alt.Chart(mdf).mark_point(shape="triangle-up", size=260, filled=True,
                                          color="#d97706", stroke="#92400e", strokeWidth=1.5)
                .encode(x="x", y="y", tooltip=["memory"])
            )

    st.altair_chart(alt.layer(*layers).interactive(), width="stretch")
    st.caption("Each dot is a chunk (colour = source file). Nearby dots mean similar "
               "meaning. The blue ◆ is your query; blue lines connect it to the chunks it "
               "retrieved (haloed, numbered by rank). Amber ▲ are long-term memories "
               "(MemoryStore) projected into the same space. **Click a file in the legend** "
               "to spotlight its chunks; hover any dot for its id, source, length, rank and "
               "similarity to your query.")


# --- session state (reflects the current on-disk index) ----------------------
ss = st.session_state
ss.setdefault("ingested_model", config.EMBED_MODEL)
ss.setdefault("chunk_size", config.CHUNK_SIZE)
ss.setdefault("overlap", config.CHUNK_OVERLAP)
ss.setdefault("fixed_chunk", False)
ss.setdefault("history", [])  # [{"query", "hits", "query_vec"}]
ss.setdefault("search_history", load_search_history())  # past queries, persisted to disk
# Assistant (full RAG chatbot) state — session memory, long-term memory, transcript.
ss.setdefault("session_mem", memory.SessionMemory(window=6))
ss.setdefault("mem_store", memory.MemoryStore())   # per-joinee, loaded on first query
ss.setdefault("mem_owner", None)                    # which joinee ss.mem_store belongs to
ss.setdefault("chat", [])  # [{"role", "content", "sources"?, "recalled"?, "retriever"?}]

# Live status of the exercise-backed controls (updates as participants implement them).
R1_DONE, R2_DONE = exercise_status()


# --- sidebar: ingestion controls ---------------------------------------------
with st.sidebar:
    st.header("⚙️ Ingestion settings")
    model = st.selectbox(
        "Embedding model", MODEL_CHOICES,
        index=MODEL_CHOICES.index(ss.ingested_model) if ss.ingested_model in MODEL_CHOICES else 0,
    )
    chunk_size = st.slider("Chunk size (chars)", 100, 2000, int(ss.chunk_size), 50)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, int(ss.overlap), 10)
    fixed = st.checkbox(
        "Use naive fixed-size chunking (to compare)", value=ss.fixed_chunk,
        help="Default is boundary-aware chunking (keeps words whole). Tick this to fall "
             "back to the naive fixed-size splitter and compare how retrieval changes.",
    )
    st.caption("Chunker: **fixed-size (naive)**" if fixed else "Chunker: **boundary-aware (default)**")

    pending = (model, chunk_size, overlap, fixed) != (
        ss.ingested_model, ss.chunk_size, ss.overlap, ss.fixed_chunk)
    if pending:
        st.caption("⚠️ Settings changed — re-ingest to apply them.")

    if st.button("🔁 Re-ingest data", type="primary", width="stretch"):
        reingest(chunk_size, overlap, model, fixed)

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
        st.write(f"**Chunker:** {'fixed-size (naive)' if ss.fixed_chunk else 'boundary-aware'}")
    else:
        st.warning("No index yet — click **Re-ingest data**.")


# --- main --------------------------------------------------------------------
st.title("🔎 Retrieval Lab")
st.caption("Play with the ingestion → retrieval flow — local embeddings + vector search, no API key.")

tab_retrieve, tab_assistant, tab_map = st.tabs(
    ["🔎 Retrieve", "💬 Assistant", "🗺️ Vector map"]
)

with tab_retrieve:
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
            "query", placeholder="Ask the knowledge base…  e.g. what payment providers do we use?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔎  Search", type="primary")

    if submitted and prompt:
        run_search(prompt)

    # Persistent search history — pick a past query to run it again.
    if ss.search_history:
        hc1, hc2, hc3 = st.columns([4, 1, 1])
        picked = hc1.selectbox(
            "Recent searches", ss.search_history, index=None,
            placeholder="⏱ Pick a past search to run again…",
            key="search_history_pick", label_visibility="collapsed",
        )
        if hc2.button("🔁 Run", width="stretch", disabled=picked is None):
            run_search(picked)
        if hc3.button("🗑 Clear", width="stretch", help="Erase the saved search history"):
            ss.search_history = []
            save_search_history([])
            st.rerun()

    if ss.history and st.button("🧹 Clear results"):
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

with tab_assistant:
    st.caption(
        "The full RAG chatbot — retrieval grounds a cited answer from the model via the "
        "**LiteLLM gateway**, with **session memory** (recent turns) and **long-term "
        "memory** (recall by meaning) wired in. Needs `LITELLM_PROXY_*` in `.env`."
    )

    c1, c3, c4 = st.columns([1.5, 0.8, 1.1])
    retr_name = c1.selectbox(
        "Retrieval", ["semantic_search", "hybrid_search"], key="chat_retriever",
        help="Which src/retrieve.py function grounds the answer. hybrid_search blends "
             "BM25 keyword matching with semantic search (R2).",
    )
    use_memory = c3.toggle(
        "Memory", value=True, key="chat_use_memory",
        help="On: feed recent turns (session memory) + facts recalled by meaning "
             "(long-term memory) into the prompt. Off: each question stands alone.",
    )
    c4.write("")
    if c4.button("🧹 Reset chat + memory", width="stretch"):
        ss.chat = []
        ss.session_mem = memory.SessionMemory(window=6)
        # Also wipe the persisted long-term store for the current joinee.
        mem_path(ss.get("chat_joinee_name", "sam")).unlink(missing_ok=True)
        ss.mem_store = memory.MemoryStore()
        ss.mem_owner = None
        st.rerun()

    pc1, pc2 = st.columns([1, 1])
    joinee_name = pc1.text_input(
        "You (joinee)", value="sam", key="chat_joinee_name",
        help="Long-term profile: loaded from disk for this name on every question "
             "and injected as system context (role + onboarding progress). Change the "
             "name to switch person; a new name creates a fresh saved profile.",
    )
    joinee_role = pc2.text_input(
        "Role", value="engineer", key="chat_joinee_role",
        help="Saved into the profile the first time we see this name.",
    )

    with st.form("assistant_form", clear_on_submit=True):
        msg = st.text_input(
            "message",
            placeholder="Ask the assistant…  e.g. How do I set up my local environment?",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("💬  Send", type="primary")

    if sent and msg:
        if not index_exists():
            st.error("No index yet — build it from the sidebar (**Re-ingest data**).")
        else:
            # Make sure the long-term MemoryStore loaded from disk belongs to THIS
            # joinee (switch person -> load their store). Persisted, so it survives
            # restarts — unlike session memory.
            if ss.mem_owner != who(joinee_name):
                ss.mem_store = memory.MemoryStore.load(mem_path(joinee_name))
                ss.mem_owner = who(joinee_name)
            # Wire memory: load the joinee's long-term PROFILE (persisted to disk),
            # recall relevant long-term facts, and take the recent session turns —
            # BEFORE adding this turn, so we don't echo the question back at the model.
            recalled = ss.mem_store.recall(msg, k=3) if (use_memory and ss.mem_store.items) else []
            prof_facts = (profile_lines(load_or_create_profile(joinee_name, joinee_role))
                          if use_memory else [])
            # Keep the two kinds separate so they show as distinct system messages in
            # the context window: the profile (who they are) vs. facts recalled by meaning.
            history = ss.session_mem.as_messages() if use_memory else []
            try:
                with st.spinner("Retrieving context and generating a grounded answer…"):
                    result = rag.answer(
                        msg, st.session_state.topk,
                        history=history, memories=prof_facts, recalled=recalled,
                        retriever=get_retriever(retr_name),
                        generate=llm.complete,
                    )
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"Generation failed: {type(e).__name__}: {e}\n\n"
                    "The LiteLLM gateway needs the proxy — check "
                    "`LITELLM_PROXY_API_BASE` / `LITELLM_PROXY_API_KEY` in `.env` "
                    "(verify with `python -m checks.check_llm`)."
                )
            else:
                # Persist the exchange so the next turn has context.
                ss.session_mem.add("user", msg)
                ss.session_mem.add("assistant", result["answer"])
                if use_memory:
                    ss.mem_store.remember(msg)              # recallable by meaning later
                    ss.mem_store.save(mem_path(joinee_name))  # persist to disk (permanent)
                ss.chat.append({
                    "query": msg, "answer": result["answer"],
                    "sources": result["sources"], "recalled": recalled,
                    "retriever": retr_name, "generator": GEN_GATEWAY,
                    "messages": result.get("messages", []),
                })
                st.rerun()

    with st.expander(f"🧠 Memory  ·  session: {len(ss.session_mem.turns)} turns  ·  "
                     f"long-term: {len(ss.mem_store.items)} facts"):
        st.caption("Session memory = the recent turns fed back on every question (sliding "
                   "window), lost when the app restarts. Long-term memory = past user "
                   "messages recalled by meaning, saved to disk per joinee "
                   f"(`{mem_path(ss.get('chat_joinee_name', 'sam')).name}`) so it survives restarts.")
        win = ss.session_mem.as_messages()
        if win:
            st.markdown("**Session window (most recent turns):**")
            for t in win:
                st.markdown(f"- _{t['role']}_: {html.escape(t['content'][:120])}")
        else:
            st.caption("No session turns yet — ask a question to start the conversation.")

        st.markdown("**Long-term profile (loaded from disk each question):**")
        if use_memory:
            for line in profile_lines(load_or_create_profile(joinee_name, joinee_role)):
                st.markdown(f"- {html.escape(line)}")
            st.caption("Injected as a system message on every question — see it in "
                       "each answer's **Context window sent to the LLM** expander.")
        else:
            st.caption("Memory is off — the profile is not injected.")

    st.write("")
    if not ss.chat:
        st.markdown(
            '<div style="opacity:.55; text-align:center; padding:32px 0; font-size:15px;">'
            "Ask the assistant a question — it retrieves context, answers from it with "
            "citations, and remembers the conversation.</div>",
            unsafe_allow_html=True,
        )
    for turn in reversed(ss.chat):  # newest exchange first, right under the input
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            meta = f"⚙ {turn['retriever']} · 🤖 {turn.get('generator', '')}"
            if turn.get("recalled"):
                meta += "  ·  🧠 recalled: " + " · ".join(m[:40] for m in turn["recalled"])
            st.caption(meta)
            with st.expander(f"Sources ({len(turn['sources'])}) · via {turn['retriever']}"):
                for i, hit in enumerate(turn["sources"], start=1):
                    st.markdown(f"**[{i}] {hit['source']}** — score {hit['score']:.2f}")
                    st.code(hit["text"][:400] + ("…" if len(hit["text"]) > 400 else ""))
            payload = turn.get("messages", [])
            with st.expander(f"📤 Context window sent to the LLM ({len(payload)} messages)"):
                render_context_window(payload)

with tab_map:
    if not index_exists():
        st.info("Re-ingest to build the vector store, then come back to see the map.")
    else:
        mem_items = ss.mem_store.items if ss.get("mem_store") else None
        if ss.history:
            last = ss.history[0]  # newest-first
            render_map(np.array(last["query_vec"]), last["query"], last["hits"],
                       memories=mem_items)
        else:
            render_map(memories=mem_items)
            st.caption("Ask a query on the Retrieve tab to see it projected onto the map.")
