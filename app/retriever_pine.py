# app/retriever_pine.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import hashlib
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

DIM = settings.embedding_dim  # 384 for MiniLM


def _stable_ascii_id(md: Dict[str, Any], fallback_idx: int) -> str:
    """
    Deterministic, ASCII-only ID based on url/anchor/source/position.
    Works even if some fields are missing.
    """
    url = str(md.get("url") or "")
    anchor = str(md.get("anchor") or "")
    source = str(md.get("source") or "")
    title = str(md.get("title") or "")
    position = str(md.get("position") if md.get("position") is not None else fallback_idx)
    raw = "|".join([url, anchor, source, title, position])
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    # host or source prefix for readability
    prefix = (source or "doc").replace(" ", "-")[:24]
    return f"{prefix}|{h}"


class PineconeRetriever:
    def __init__(self):
        # --- Embeddings model ---
        self.embedder = SentenceTransformer(settings.embedding_model_name, device="cpu")
        try:
            self.embedder.max_seq_length = 512
        except Exception:
            pass

        # --- Pinecone client ---
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        # --- Ensure index exists (serverless) ---
        name = settings.pinecone_index
        names = {i["name"] for i in self.pc.list_indexes()}
        if name not in names:
            # Cloud/region from config (serverless)
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
        # Avoid NumPy requirement: get torch tensors and turn into Python lists
        vecs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_tensor=True,   # <- returns a single torch.Tensor
            show_progress_bar=False,
        )
        # vecs is shape (N, D) torch.Tensor
        return vecs.detach().cpu().tolist()

    # -------- Upsert (chunks) --------
    def upsert_chunks(self, chunks: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        chunks: [{ "text": "...", "metadata": {...} }, ...]
        We KEEP rich metadata so the UI can make url#anchor links:
        url, anchor, title, section / section_heading, position, source, product, version, source_type.
        """
        namespace = namespace or settings.pinecone_namespace

        if not chunks:
            return

        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embed(texts)

        vectors = []
        for i, (c, vec) in enumerate(zip(chunks, embeddings)):
            md_in = (c.get("metadata") or {}).copy()

            # Normalize some common fields and keep them
            # Prefer section_heading if provided
            section = md_in.get("section") or md_in.get("section_heading")
            md_out = {
                "text": c.get("text", ""),
                "url": md_in.get("url"),
                "anchor": md_in.get("anchor"),
                "title": md_in.get("title"),
                "section": section,
                "section_heading": md_in.get("section_heading"),
                "position": md_in.get("position"),
                "source": md_in.get("source"),
                "product": md_in.get("product"),
                "version": md_in.get("version"),
                "source_type": md_in.get("source_type"),
                # optional short preview for UI
                "text_preview": c.get("text", "")[:320],
            }

            vid = _stable_ascii_id(md_out, i)
            vectors.append({"id": vid, "values": vec, "metadata": md_out})

        # Upsert in batches (v5)
        B = 200
        for j in range(0, len(vectors), B):
            batch = vectors[j : j + B]
            self.index.upsert(vectors=batch, namespace=namespace)

    # -------- Retrieve (vector search) --------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
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


