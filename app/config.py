# app/config.py
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Optional: load config/.env if you ever add one
load_dotenv(dotenv_path=Path("config/.env"))

# ---- secrets loader (supports flat and nested tables) ----
def _flatten(d, parent_key=""):
    flat = {}
    for k, v in d.items():
        key = f"{parent_key}{k}".upper() if not parent_key else f"{parent_key}_{k}".upper()
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat

def load_streamlit_secrets():
    """Load keys from .streamlit/secrets.toml into os.environ if present (handles nested tables)."""
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # for <=3.10
        except Exception:
            return

    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        return

    with open(secrets_path, "rb") as f:
        data = tomllib.load(f)

    flat = _flatten(data)
    for key, value in flat.items():
        if key not in os.environ and value is not None:
            os.environ[key] = str(value)

# load once on import
load_streamlit_secrets()
# ----------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-mini")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")

    # Embeddings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # Reranker (accept COHERE_API_KEY or CO_API_KEY)
    cohere_api_key: str = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY", "")
    cohere_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    initial_recall_k: int = int(os.getenv("INITIAL_RECALL_K", "25"))

    # LLM (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Chunking / retrieval
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))
    chunk_overlap: float = float(os.getenv("CHUNK_OVERLAP", "0.12"))
    max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "6"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))

# >>> IMPORTANT: expose a module-level settings object <<<
settings = Settings()
