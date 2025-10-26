# app/pipeline.py
from typing import List, Dict, Any, Optional
import time
import cohere

from app.config import settings
from app.retriever_pine import PineconeRetriever
from app.llm import GroqLLM, SYSTEM_PROMPT
from app.utils import build_inline_citations, insert_citation_tags, clean_text, mmr


class RagPipeline:
    def __init__(self):
        self.retriever = PineconeRetriever()
        self.llm = GroqLLM()
        self.cohere = cohere.Client(api_key=settings.cohere_api_key) if settings.cohere_api_key else None
        self.rerank_model = settings.cohere_model

    # ---------------------------
    # Ingest (kept simple/on-demand)
    # ---------------------------
    def ingest_document(
        self,
        text: str,
        source: str,
        title: str = "",
        section: str = "",
        namespace: Optional[str] = None,
    ):
        from app.utils import sliding_window_chunk
        chunks = sliding_window_chunk(
            text=text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_ratio=settings.chunk_overlap,
            meta={"source": source, "title": title, "section": section},
        )
        # Respect namespace if provided; retriever will default otherwise
        self.retriever.upsert_chunks(chunks, namespace=namespace)

    # ---------------------------
    # Retrieval + (MMR) + Rerank
    # ---------------------------

    def retrieve_and_rerank(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Dense -> broaden recall (if keywordy) -> keyword filter -> MMR -> Cohere rerank (rich docs)
        Returns {'hits': [...], 'timings': {...}, 'rerank_used': bool}
        """
        t_retrieve = 0.0
        t_rerank = 0.0

        # ---- Keyword heuristics for this query ----
        qlow = (query or "").lower()
        KEYWORDS = (
            " with ", "with statement", "context manager", "contextlib", "open(",
            "file object", "closing files", "try/finally"
        )
        has_kw = any(kw in qlow for kw in KEYWORDS)

        # If keywordy, ensure broad recall; otherwise use provided/default k
        base_k = top_k or settings.initial_recall_k
        k = max(base_k, 80) if has_kw else base_k

        # ---- Dense retrieval (broad recall; min_score=0) ----
        t0 = time.time()
        try:
            initial_hits = self.retriever.retrieve(
                query,
                top_k=k,
                namespace=namespace,
                min_score=0.0,  # broaden recall; we'll filter ourselves
            )
        except Exception as e:
            print(f"[ERROR] Pinecone retrieve failed: {e}")
            return {"hits": [], "timings": {"retrieve_s": 0.0, "rerank_s": 0.0}, "rerank_used": False}
        t_retrieve = time.time() - t0

        if not initial_hits:
            return {"hits": [], "timings": {"retrieve_s": t_retrieve, "rerank_s": 0.0}, "rerank_used": False}

        # ---- Keyword filter (require at least one hit contain the topic) ----
        def _kw_hit(d: Dict[str, Any]) -> bool:
            md = d.get("metadata") or {}
            hay = f"{md.get('title','')} {(md.get('section') or md.get('section_heading') or '')} {d.get('text','')}".lower()
            if any(kw in hay for kw in KEYWORDS):
                return True
            url = (md.get("url") or "").lower()
            if any(tok in url for tok in ("with-statement", "contextlib", "compound_stmts", "io.html")):
                return True
            return False

        filtered = [d for d in initial_hits if _kw_hit(d)]
        pool = filtered if filtered else initial_hits  # fall back if nothing matched

        # ---- Hard boost: bubble exact ref pages earlier in pool ----
        def _hard_boost_rank(d: Dict[str, Any]) -> float:
            url = (d.get("metadata", {}).get("url") or "").lower()
            boost = 0.0
            if "compound_stmts.html" in url and "with" in url:
                boost += 5.0
            if "contextlib" in url:
                boost += 3.0
            return boost

        pool.sort(key=_hard_boost_rank, reverse=True)

        # ---- MMR diversify on the pool ----
        try:
            embs = self.retriever.embed([h["text"] for h in pool])
            mmr_idx = mmr(embs, top_k=min(24, len(pool)), lambda_mult=0.55)
            diversified = [pool[i] for i in mmr_idx]
        except Exception as e:
            print(f"[WARN] MMR failed, using top-{min(24, len(pool))}: {e}")
            diversified = pool[: min(24, len(pool))]

        # ---- Prepare richer docs for Cohere Rerank ----
        docs_for_rerank = []
        for d in diversified:
            md = d.get("metadata") or {}
            title = md.get("title") or md.get("source") or ""
            section = md.get("section") or md.get("section_heading") or ""
            url = md.get("url") or ""
            docs_for_rerank.append(f"Title: {title}\nSection: {section}\nURL: {url}\n\n{d.get('text','')}")

        # ---- Cohere Rerank (fallback to diversified dense) ----
        try:
            if not self.cohere:
                raise RuntimeError("COHERE_API_KEY not set")
            t1 = time.time()
            rr = self.cohere.rerank(
                model=self.rerank_model,
                query=query,
                documents=docs_for_rerank,
                top_n=min(rerank_top_k or settings.rerank_top_k, len(docs_for_rerank)),
            )
            t_rerank = time.time() - t1

            reranked = []
            for r in rr.results:
                item = diversified[r.index]
                reranked.append({**item, "rerank_score": r.relevance_score})

            # tiny tie-breaker nudge by keyword signal
            def _signal(d):
                md = d.get("metadata") or {}
                hay = f"{md.get('title','')} {(md.get('section') or md.get('section_heading') or '')} {d.get('text','')}".lower()
                s = 0.0
                for kw in KEYWORDS:
                    if kw in hay:
                        s += 0.05
                return s

            reranked.sort(key=lambda d: (-(d.get("rerank_score") or 0.0), -_signal(d)))
            return {
                "hits": reranked,
                "timings": {"retrieve_s": t_retrieve, "rerank_s": t_rerank},
                "rerank_used": True,
            }
        except Exception as e:
            print(f"[WARN] Cohere rerank failed or skipped; using diversified dense: {e}")
            topn = min(rerank_top_k or settings.rerank_top_k, len(diversified))
            return {
                "hits": diversified[:topn],
                "timings": {"retrieve_s": t_retrieve, "rerank_s": 0.0},
                "rerank_used": False,
            }





    # ---------------------------
    # Answer with citations
    # ---------------------------
    def answer(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        rr = self.retrieve_and_rerank(query, namespace=namespace, top_k=top_k, rerank_top_k=rerank_top_k)
        reranked = rr["hits"]
        timings = rr["timings"]

        if not reranked:
            ns_msg = f" (namespace='{namespace}')" if namespace else ""
            return {
                "answer": f"I couldn’t find enough information in your indexed docs{ns_msg}. "
                        f"Try a different namespace or a simpler query.",
                "contexts": [],
                "sources": [],
                "metrics": {
                    "retrieve_s": timings.get("retrieve_s", 0.0),
                    "rerank_s": timings.get("rerank_s", 0.0),
                    "retriever_ms": int(timings.get("retrieve_s", 0.0) * 1000),
                    "rerank_ms": int(timings.get("rerank_s", 0.0) * 1000),
                    "llm_latency_s": 0.0,
                    "llm_ms": 0,
                    "tokens_input": None,
                    "tokens_output": None,
                    "model": settings.groq_model,
                    "rerank_used": rr.get("rerank_used", False),
                },
            }


        # Take the best N for context
        contexts = reranked[: settings.max_context_docs]

        # --- Build inline citation numbering ---
        # Normalize contexts for numbering; prefer section_heading if present
        norm_ctx: List[Dict[str, Any]] = []
        for c in contexts:
            md = c.get("metadata") or {}
            section = md.get("section") or md.get("section_heading")
            norm_ctx.append({
                "text": c.get("text", ""),
                "source": md.get("source"),
                "title": md.get("title"),
                "section": section,
                "position": md.get("position"),
                "url": md.get("url"),
                "anchor": md.get("anchor"),
            })

        _, unique_sources = build_inline_citations(norm_ctx)

        # Map (source, title, section, position) -> [n]
        key_to_num: Dict[Any, int] = {}
        for i, s in enumerate(unique_sources, start=1):
            key = (s.get("source"), s.get("title"), s.get("section"), s.get("position"))
            key_to_num[key] = i

        # Attach cite numbers to original contexts
        for c in contexts:
            md = c.get("metadata") or {}
            key = (
                md.get("source"),
                md.get("title"),
                md.get("section") or md.get("section_heading"),
                md.get("position"),
            )
            c["cite_num"] = key_to_num.get(key, "?")

        # --- Build user-visible Sources with clickable links ---
        display_sources: List[Dict[str, Any]] = []
        for c in contexts:
            md = c.get("metadata") or {}
            title = md.get("title") or md.get("source") or "Source"
            section = md.get("section") or md.get("section_heading")
            url = md.get("url")
            anchor = md.get("anchor")
            link = f"{url}#{anchor}" if (url and anchor) else url
            display_sources.append({
                "n": c.get("cite_num", "?"),
                "source": md.get("source"),
                "title": title,
                "section": section,
                "position": md.get("position"),
                "url": url,
                "anchor": anchor,
                "link": link,
                "snippet": c.get("text", "")[:300] + ("..." if len(c.get("text", "")) > 300 else ""),
            })
        display_sources.sort(key=lambda x: (9999 if x["n"] == "?" else int(x["n"])))

        # A. Inline-tagged excerpt block (what you already have)
        context_block = insert_citation_tags(contexts)

        # B. Add a brief sources summary (helps the model key in before reading chunks)
        #    Example line: [1] 8.5. The with statement — https://docs.python.org/...#the-with-statement
        src_lines = []
        for s in sorted([{
                "n": c.get("cite_num", "?"),
                "title": (c.get("metadata") or {}).get("title") or (c.get("metadata") or {}).get("source") or "Source",
                "section": (c.get("metadata") or {}).get("section") or (c.get("metadata") or {}).get("section_heading"),
                "url": (c.get("metadata") or {}).get("url"),
            } for c in contexts], key=lambda x: (9999 if x["n"] == "?" else int(x["n"]))):
            t = s["title"]
            if s["section"]:
                t = f"{t} — {s['section']}"
            line = f"[{s['n']}] {t}"
            if s["url"]:
                line += f" — {s['url']}"
            src_lines.append(line)

        sources_summary = "Sources:\n" + "\n".join(src_lines) if src_lines else "Sources:\n(none)"

        # C. Strong guardrails: if we have any context, the model MUST use it and must not claim insufficiency.
        has_any = len(contexts) > 0
        use_context_hint = (
            "You DO have relevant documentation excerpts. Use them. "
            "Do NOT say the context is insufficient. If something is not in the context, say so briefly and continue using what is provided."
            if has_any else
            "If the provided context truly does not address the question, briefly say so."
        )

        SYSTEM_RULES = (
            "Rules:\n"
            "- Use only the provided Context unless the answer is trivial Python knowledge.\n"
            "- Ground every claim with inline citations like [1], [2] that refer to the Context items.\n"
            "- Do NOT claim the context is insufficient if any context is provided.\n"
            "- Prefer clear, practical explanations and a minimal runnable code example when applicable."
        )

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\n{SYSTEM_RULES}",
            },
            {
                "role": "user",
                "content": (
                    f"{use_context_hint}\n\n"
                    f"Question: {query}\n\n"
                    f"{sources_summary}\n\n"
                    f"Context excerpts (each line may end with a [n] citation tag):\n{context_block}\n\n"
                    "Answer concisely (5–8 lines). Include a short code example if relevant. "
                    "Add inline citations [n] in your answer where you use a fact."
                ),
            },
        ]

        # Generate with meta if available
        model_name = settings.groq_model
        latency_s = 0.0
        tokens_in = None
        tokens_out = None

        if hasattr(self.llm, "generate_with_meta"):
            llm_res = self.llm.generate_with_meta(messages)
            final_answer = clean_text(llm_res.get("text", ""))
            model_name = llm_res.get("model", model_name)
            latency_s = llm_res.get("latency_s", 0.0)
            usage = llm_res.get("usage") or {}
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")
        else:
            t0 = time.time()
            final_answer = clean_text(self.llm.generate(messages))
            latency_s = time.time() - t0

        # --- Return ---
        return {
            "answer": final_answer,
            "contexts": contexts,
            "sources": display_sources,
            "metrics": {
                # seconds
                "retrieve_s": timings.get("retrieve_s", 0.0),
                "rerank_s": timings.get("rerank_s", 0.0),
                "llm_latency_s": latency_s,
                # ms
                "retriever_ms": int(timings.get("retrieve_s", 0.0) * 1000),
                "rerank_ms": int(timings.get("rerank_s", 0.0) * 1000),
                "llm_ms": int(latency_s * 1000),
                # tokens + model info
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "model": model_name,
                "rerank_used": rr.get("rerank_used", False),
            },
        }
