# app/pipeline.py
from typing import List, Dict, Any
from app.config import settings
from app.retriever_pine import PineconeRetriever
from app.llm import GroqLLM, SYSTEM_PROMPT
from app.utils import build_inline_citations, insert_citation_tags, clean_text, mmr
import cohere
import time

class RagPipeline:
    def __init__(self):
        self.retriever = PineconeRetriever()
        self.llm = GroqLLM()
        self.cohere = cohere.Client(api_key=settings.cohere_api_key)
        self.rerank_model = settings.cohere_model

    # ... keep ingest_document as-is ...

    def retrieve_and_rerank(self, query: str) -> Dict[str, Any]:
        """Always returns a dict: {'hits': [...], 'timings': {'retrieve_s': float, 'rerank_s': float}, 'rerank_used': bool}"""
        t_retrieve = 0.0
        t_rerank = 0.0
        try:
            t0 = time.time()
            # Dense retrieval
            initial_hits = self.retriever.retrieve(query, top_k=settings.initial_recall_k)
            t_retrieve = time.time() - t0

            if not initial_hits:
                return {"hits": [], "timings": {"retrieve_s": t_retrieve, "rerank_s": 0.0}, "rerank_used": False}

            # MMR diversify (optional)
            embs = self.retriever.embed([h["text"] for h in initial_hits])
            from app.utils import mmr
            mmr_idx = mmr(embs, top_k=min(12, len(initial_hits)), lambda_mult=0.55)
            diversified = [initial_hits[i] for i in mmr_idx]

            # Cohere rerank
            try:
                t1 = time.time()
                rr = self.cohere.rerank(
                    model=self.rerank_model,
                    query=query,
                    documents=[d["text"] for d in diversified],
                    top_n=min(settings.rerank_top_k, len(diversified)),
                )
                t_rerank = time.time() - t1

                reranked = []
                for r in rr.results:
                    item = diversified[r.index]
                    reranked.append({**item, "rerank_score": r.relevance_score})
                return {"hits": reranked, "timings": {"retrieve_s": t_retrieve, "rerank_s": t_rerank}, "rerank_used": True}
            except Exception as e:
                # Fallback to dense retrieval if rerank fails
                print(f"[WARN] Cohere rerank failed, using dense retrieval only: {e}")
                reranked = diversified[: min(settings.rerank_top_k, len(diversified))]
                return {"hits": reranked, "timings": {"retrieve_s": t_retrieve, "rerank_s": 0.0}, "rerank_used": False}

        except Exception as e:
            # Any unexpected failure -> safe empty result
            print(f"[ERROR] retrieve_and_rerank crashed: {e}")
            return {"hits": [], "timings": {"retrieve_s": t_retrieve, "rerank_s": t_rerank}, "rerank_used": False}



    def answer(self, query: str) -> Dict[str, Any]:
        # Retrieve + rerank with timings
        rr = self.retrieve_and_rerank(query)
        reranked = rr["hits"]
        timings = rr["timings"]

        if not reranked:
            return {
                "answer": "I couldn’t find enough information in your documents to answer that confidently.",
                "contexts": [],
                "sources": [],
                "metrics": {
                    "retrieve_s": timings["retrieve_s"],
                    "rerank_s": timings["rerank_s"],
                    "llm_latency_s": 0.0,
                    "llm_tokens": None,
                    "model": settings.groq_model,
                },
            }

        contexts = reranked[: settings.max_context_docs]

        # Assign citation numbers
        _, unique_sources = build_inline_citations([{
            "text": c["text"],
            "source": c["metadata"].get("source"),
            "title": c["metadata"].get("title"),
            "section": c["metadata"].get("section"),
            "position": c["metadata"].get("position"),
        } for c in contexts])

        key_to_num = {}
        for i, s in enumerate(unique_sources, start=1):
            key_to_num[(s.get("source"), s.get("title"), s.get("section"), s.get("position"))] = i
        for c in contexts:
            key = (
                c["metadata"].get("source"),
                c["metadata"].get("title"),
                c["metadata"].get("section"),
                c["metadata"].get("position"),
            )
            c["cite_num"] = key_to_num.get(key, "?")

        context_block = insert_citation_tags(contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer with inline citations like [1], [2]."
            },
        ]

        # LLM call with meta 
        if hasattr(self.llm, "generate_with_meta"):
            llm_res = self.llm.generate_with_meta(messages)
            final_answer = clean_text(llm_res["text"])
            model_name = llm_res.get("model", settings.groq_model)
            latency_s = llm_res.get("latency_s", 0.0)
            usage = llm_res.get("usage")
        else:
            t0 = time.time()
            final_answer = clean_text(self.llm.generate(messages))
            latency_s = time.time() - t0
            model_name = settings.groq_model
            usage = None

        display_sources = []
        for c in contexts:
            md = c["metadata"]
            display_sources.append({
                "n": c["cite_num"],
                "source": md.get("source"),
                "title": md.get("title"),
                "section": md.get("section"),
                "position": md.get("position"),
                "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else "")
            })
        display_sources.sort(key=lambda x: x["n"])

        return {
        "answer": final_answer,
        "contexts": contexts,
        "sources": display_sources,
        "metrics": {
            "retrieve_s": timings["retrieve_s"],
            "rerank_s": timings["rerank_s"],
            "llm_latency_s": latency_s,
            "llm_tokens": usage,
            "model": model_name,
            "rerank_used": rr.get("rerank_used", False),  # <-- add this
        },
    }
    
    def ingest_document(self, text: str, source: str, title: str = "", section: str = ""):
        from app.utils import sliding_window_chunk
        from app.config import settings
        chunks = sliding_window_chunk(
            text=text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_ratio=settings.chunk_overlap,
            meta={"source": source, "title": title, "section": section},
        )
        self.retriever.upsert_chunks(chunks)

