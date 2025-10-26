# app/utils.py
from typing import List, Dict, Any, Tuple
import math
import re

def approximate_token_len(text: str) -> int:
    # Rough token estimate ~ words * 1.3 (quick & dependency-free)
    words = len(text.split())
    return int(words * 1.3)

def sliding_window_chunk(
    text: str,
    chunk_size_tokens: int = 1000,
    overlap_ratio: float = 0.12,
    meta: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Chunk text to ~chunk_size_tokens with ~overlap_ratio overlap.
    Store metadata for citations.
    """
    meta = meta or {}
    words = text.split()
    approx_tokens_per_word = 1.3
    words_per_chunk = max(50, int(chunk_size_tokens / approx_tokens_per_word))
    overlap_words = int(words_per_chunk * overlap_ratio)

    chunks = []
    start = 0
    position = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "metadata": {
                **meta,
                "position": position,
                "char_count": len(chunk_text),
            }
        })
        position += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks

def build_inline_citations(sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Map unique (source,title,section,position) to [n] indices.
    Returns citation string for prompt and the ordered unique source list.
    """
    unique = {}
    ordered = []
    for s in sources:
        key = (s.get("source"), s.get("title"), s.get("section"), s.get("position"))
        if key not in unique:
            unique[key] = len(unique) + 1
            ordered.append(s)
        s["cite_num"] = unique[key]

    # Build a compact reference block (you’ll show this under the answer in UI)
    # Not used in the LLM prompt; prompt uses bracket tags inline.
    return "", ordered

def insert_citation_tags(contexts: List[Dict[str, Any]]) -> str:
    """
    Compose a retrieval context string with inline [n] tags.
    """
    blocks = []
    for c in contexts:
        n = c.get("cite_num", "?")
        txt = c["text"].strip()
        src = c["metadata"].get("source", "unknown")
        blocks.append(f"[{n}] {txt}\n(Source: {src})")
    return "\n\n".join(blocks)

def mmr(
    embeddings: list[list[float]],
    top_k: int,
    lambda_mult: float = 0.5
) -> list[int]:
    """
    Simple MMR index selection over a list of embeddings (cosine).
    You can apply this on the initial Pinecone hits to promote diversity.
    """
    import numpy as np
    E = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms

    selected = []
    candidates = set(range(E.shape[0]))
    # assume first doc is best per initial score
    current = 0
    selected.append(current)
    candidates.remove(current)

    while len(selected) < min(top_k, E.shape[0]):
        # relevance to query approximated by first vector (0)
        relevance = E[0] @ E.T
        max_div_score, next_idx = -1e9, None
        for j in candidates:
            diversity = max(E[j] @ E[selected].T)
            score = lambda_mult * relevance[j] - (1 - lambda_mult) * diversity
            if score > max_div_score:
                max_div_score, next_idx = score, j
        selected.append(next_idx)
        candidates.remove(next_idx)
    return selected

def clean_text(s: str) -> str:
    # Light clean to avoid prompt injection helpfully (still keep raw for citations panel)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()
