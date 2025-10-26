# streamlit_app.py
import streamlit as st
from app import token_tracker
from app.pipeline import RagPipeline

# -------- Helpers --------
def _extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts text from an uploaded PDF using PyPDF2 if available.
    Falls back to a naive byte decode if PDF parsing lib is missing.
    """
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(uploaded_file)
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n\n".join(texts).strip()
    except Exception:
        try:
            data = uploaded_file.getvalue()
            return data.decode(errors="ignore")
        except Exception:
            return ""

# ---------------- Page config ----------------
st.set_page_config(
    page_title="MINI_RAG — Pinecone + MiniLM + Cohere + Groq",
    layout="wide",
)

# ---------------- Centered loading message until pipeline is ready ----------------
placeholder = st.empty()
with placeholder.container():
    st.markdown(
        "<h2 style='text-align:center;'>⏳ Loading… please wait</h2>",
        unsafe_allow_html=True
    )

# ---------------- Instantiate pipeline ----------------
pipe = RagPipeline()

# ---------------- Clear loading ----------------
placeholder.empty()

# ---------------- Header ----------------
st.markdown(
    "<div style='display:flex;align-items:center;gap:10px'>"
    "<h2 style='margin:0;'>🤖 MINI_RAG Chatbot</h2>"
    "<span style='opacity:0.7'>Pinecone · MiniLM · Cohere Rerank · Groq</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar: Ingestion & Options ----------------
with st.sidebar:
    st.header("📥 Ingest")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    text_to_index = st.text_area(
        "Or paste text to index",
        height=160,
        placeholder="Paste raw text here…"
    )

    can_ingest = (pdf_file is not None) or (text_to_index.strip() != "")
    ingest_btn = st.button(
        "Ingest",
        type="primary",
        use_container_width=True,
        disabled=not can_ingest
    )

    if ingest_btn:
        # PDF
        if pdf_file is not None:
            with st.spinner("Reading PDF and chunking…"):
                pdf_text = _extract_text_from_pdf(pdf_file)
            if pdf_text.strip():
                src_name = getattr(pdf_file, "name", "uploaded.pdf")
                pipe.ingest_document(pdf_text, source=src_name, title="", section="")
                st.success(f"Ingested from PDF: {src_name} ✅")
            else:
                st.error("Could not extract any text from the PDF. Please check the file.")

        # Pasted text
        if text_to_index.strip():
            pipe.ingest_document(
                text_to_index.strip(),
                source="local-paste",
                title="",
                section=""
            )
            st.success("Ingested pasted text ✅")

        if not (pdf_file is not None or text_to_index.strip()):
            st.warning("Upload a PDF or paste some text to ingest.")

    st.divider()
    st.subheader("⚙️ Options")
    auto_expand_sources = st.checkbox("Auto-expand Sources", value=True)
    auto_expand_metrics = st.checkbox("Auto-expand Metrics", value=False)
    st.caption("Tip: keep Sources open while verifying grounding.")

# ---------------- Chat state ----------------
if "chat_history" not in st.session_state:
    # each: {"query": str, "answer": str, "out": dict}
    st.session_state.chat_history = []

# Clear chat
cols = st.columns([1, 1, 6])
with cols[0]:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- Chat log ----------------
for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

        # Sources expander
        src_open = auto_expand_sources
        with st.expander("📚 Sources", expanded=src_open):
            if turn["out"].get("sources"):
                for s in turn["out"]["sources"]:
                    st.markdown(f"**[{s['n']}] {s.get('title') or s.get('source')}**")
                    small = f"{s.get('source','')} • {s.get('section','')} • pos {s.get('position')}"
                    st.caption(small)
                    st.code(s["snippet"])
            else:
                st.caption("No sources returned.")

        # Metrics expander
        met_open = auto_expand_metrics
        with st.expander("📈 Metrics", expanded=met_open):
            m = turn["out"].get("metrics", {}) or {}
            tok = m.get("llm_tokens") or {}
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("LLM latency", f"{m.get('llm_latency_s', 0):.2f}s")
            col2.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
            col3.metric("Rerank", f"{m.get('rerank_s', 0):.2f}s")
            col4.metric("Model", m.get("model", "—"))

            total_used = tok.get("total_tokens")
            if total_used:
                left, used, limit = token_tracker.add_tokens(int(total_used))
                col5.metric("Tokens left (today)", f"{left:,}", f"-{used:,}/{limit:,}")
                st.caption(
                    f"Tokens — Prompt: {tok.get('prompt_tokens','?')}, "
                    f"Completion: {tok.get('completion_tokens','?')}, "
                    f"Total: {tok.get('total_tokens','?')}"
                )
            else:
                col5.metric("Tokens left (today)", "—")
                st.caption("Tokens — not returned by provider for this response.")

# ---------------- New message ----------------
q = st.chat_input("Ask your question…")
if q and q.strip():
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            out = pipe.answer(q)

        st.write(out["answer"])

        # Sources expander
        src_open = auto_expand_sources
        with st.expander("📚 Sources", expanded=src_open):
            if out.get("sources"):
                for s in out["sources"]:
                    st.markdown(f"**[{s['n']}] {s.get('title') or s.get('source')}**")
                    small = f"{s.get('source','')} • {s.get('section','')} • pos {s.get('position')}"
                    st.caption(small)
                    st.code(s["snippet"])
            else:
                st.caption("No sources returned.")

        # Metrics expander
        met_open = auto_expand_metrics
        with st.expander("📈 Metrics", expanded=met_open):
            m = out.get("metrics", {}) or {}
            tok = m.get("llm_tokens") or {}
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("LLM latency", f"{m.get('llm_latency_s', 0):.2f}s")
            col2.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
            col3.metric("Rerank", f"{m.get('rerank_s', 0):.2f}s")
            col4.metric("Model", m.get("model", "—"))

            total_used = tok.get("total_tokens")
            if total_used:
                left, used, limit = token_tracker.add_tokens(int(total_used))
                col5.metric("Tokens left (today)", f"{left:,}", f"-{used:,}/{limit:,}")
                st.caption(
                    f"Tokens — Prompt: {tok.get('prompt_tokens','?')}, "
                    f"Completion: {tok.get('completion_tokens','?')}, "
                    f"Total: {tok.get('total_tokens','?')}"
                )
            else:
                col5.metric("Tokens left (today)", "—")
                st.caption("Tokens — not returned by provider for this response.")

    # persist
    st.session_state.chat_history.append(
        {"query": q, "answer": out["answer"], "out": out}
    )
