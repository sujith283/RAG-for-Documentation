# app/retriever_pine.py
from __future__ import annotations

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


DIM = settings.embedding_dim  # 384 for MiniLM


class PineconeRetriever:
    def __init__(self):
        # --- Embeddings model ---
        # Normalize embeddings to match cosine metric best practices.
        self.embedder = SentenceTransformer(settings.embedding_model_name)

        # --- Pinecone client ---
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        # --- Ensure index exists (serverless) ---
        name = settings.pinecone_index
        existing = [i["name"] for i in self.pc.list_indexes().get("indexes", [])]
        if name not in existing:
            # Cloud/region come from your config (e.g., cloud="aws", region="us-east-1")
            self.pc.create_index(
                name=name,
                dimension=DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region or "us-east-1",
                ),
            )

        # --- Open index handle ---
        self.index = self.pc.Index(name)

    # -------- Embeddings --------
    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return vecs.tolist() if hasattr(vecs, "tolist") else vecs

    # -------- Upsert (chunks) --------
    def upsert_chunks(self, chunks: List[Dict[str, Any]], namespace: str | None = None):
        namespace = namespace or settings.pinecone_namespace

        vectors = []
        for i, c in enumerate(chunks):
            # c: {"text": "...", "metadata": {"source": "...", "title": "...", "section": "...", "position": int}}
            values = self.embed([c["text"]])[0]
            md_in = c.get("metadata", {}) or {}
            metadata = {
                "text": c["text"],
                # keep only the keys you care about (used later for citations)
                **{k: v for k, v in md_in.items() if k in ("source", "title", "section", "position")}
            }
            vid = f'{metadata.get("source", "doc")}:{metadata.get("position", i)}'
            vectors.append({"id": vid, "values": values, "metadata": metadata})

        # Pinecone v3 upsert
        if vectors:
            self.index.upsert(vectors=vectors, namespace=namespace)

    # -------- Retrieve (vector search) --------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        min_score: float = 0.25,
    ):
        top_k = top_k or settings.initial_recall_k
        namespace = namespace or settings.pinecone_namespace

        qvec = self.embed([query])[0]
        res = self.index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )

        hits = []
        # Pinecone v3 returns dict with "matches"
        for m in res.get("matches", []):
            score = m.get("score", 0.0)
            if score >= min_score:
                md = m.get("metadata") or {}
                hits.append(
                    {
                        "id": m.get("id"),
                        "score": score,
                        "text": md.get("text", ""),
                        "metadata": md,
                    }
                )
        return hits


