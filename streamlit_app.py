# streamlit_app.py
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import streamlit as st
from app.pipeline import RagPipeline
from app.config import settings

st.set_page_config(page_title="Docs RAG — Answers with Citations", page_icon="📚", layout="wide")
st.markdown(
    "<style>.smallcaps{opacity:.75;font-size:.9rem} .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;}</style>",
    unsafe_allow_html=True,
)

# -------- Init once --------
if "pipe" not in st.session_state:
    st.session_state.pipe = RagPipeline()
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{q, out}]

# -------- Sidebar --------
with st.sidebar:
    st.title("📚 Docs RAG")
    st.caption("Answers grounded in your docs with clickable citations")

    default_ns = getattr(settings, "pinecone_namespace", None) or os.getenv("PINECONE_NAMESPACE", "")
    namespace = st.text_input(
        "Pinecone namespace",
        value=default_ns,
        placeholder="e.g., python@3.12, fastapi@0.115",
        help="Must match the namespace you ingested.",
    )
    topk = st.slider("Initial recall (k)", 10, 50, 25, 5)
    rerank_k = st.slider("Rerank top-N", 4, 20, 10, 1)
    
    st.divider()
    st.caption("Retrieval smoke test (no LLM)")
    if st.button("🔎 Test retrieval (top 3)"):
        try:
            r = st.session_state.pipe.retriever
            hits = r.retrieve("open a file with context manager", top_k=3, namespace=namespace or None, min_score=0.0)
            st.success(f"Retrieved {len(hits)} hits")
            if hits:
                st.json(hits[0].get("metadata", {}))
        except Exception as e:
            st.error("Retrieval failed.")
            st.exception(e)


    st.divider()
    with st.expander("Debug (runtime)"):
        st.write({
            "namespace (sidebar)": namespace or "(none)",
            "env:PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
            "env:PINECONE_NAMESPACE": os.getenv("PINECONE_NAMESPACE"),
            "keys_present": {
                "pinecone": bool(settings.pinecone_api_key),
                "groq": bool(settings.groq_api_key),
                "cohere": bool(getattr(settings, "cohere_api_key", "")),
            },
        })
    st.caption("Tip: Set the namespace to the docs you ingested (e.g., python@3.12).")

# -------- Helpers --------
def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    st.subheader("Sources (click to open)")
    for s in sources:
        title = s.get("title") or s.get("source") or "Source"
        link = s.get("link")
        n = s.get("n", "?")
        header = f"**[{n}] {title}**" if not link else f"**[{n}] [{title}]({link})**"
        st.markdown(header)

        parts = []
        if s.get("source"): parts.append(str(s["source"]))
        if s.get("section"): parts.append(str(s["section"]))
        if s.get("position") is not None: parts.append(f"pos {s['position']}")
        if parts: st.caption(" • ".join(parts))

        snippet = s.get("snippet", "")
        if snippet: st.code(snippet)

def render_metrics(metrics: Optional[Dict[str, Any]]):
    if not metrics:
        return
    st.subheader("Metrics")
    cols = st.columns(3)
    cols[0].metric("Retriever (ms)", metrics.get("retriever_ms", "—"))
    cols[1].metric("Rerank (ms)", metrics.get("rerank_ms", "—"))
    cols[2].metric("LLM (ms)", metrics.get("llm_ms", "—"))
    with st.expander("More"):
        st.json({
            "model": metrics.get("model"),
            "tokens_input": metrics.get("tokens_input"),
            "tokens_output": metrics.get("tokens_output"),
            "retrieve_s": metrics.get("retrieve_s"),
            "rerank_s": metrics.get("rerank_s"),
            "llm_latency_s": metrics.get("llm_latency_s"),
            "total_ms": metrics.get("total_ms"),
            "rerank_used": metrics.get("rerank_used"),
        })

# -------- History --------
st.markdown("## Ask the docs")
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["out"].get("answer", ""))
        render_sources(turn["out"].get("sources", []))
        render_metrics(turn["out"].get("metrics", {}))

# -------- Input / Call --------
prompt = st.chat_input("Ask a question about the indexed docs…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                out = st.session_state.pipe.answer(
                    query=prompt,
                    namespace=namespace or None,
                    top_k=topk,
                    rerank_top_k=rerank_k,
                )
            except Exception as e:
                st.error("The pipeline raised an exception.")
                st.exception(e)
                st.text("Traceback:")
                st.code(traceback.format_exc())
                out = {"answer": "", "sources": [], "metrics": {}}
            t1 = time.time()

        # Render
        st.markdown(out.get("answer", ""))
        render_sources(out.get("sources", []))

        # add total time if not present
        metrics = out.get("metrics", {}) or {}
        metrics.setdefault("total_ms", int((t1 - t0) * 1000))
        render_metrics(metrics)

    st.session_state.chat.append({"q": prompt, "out": out})
