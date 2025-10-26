# app/llm.py
from typing import List, Dict, Any
from app.config import settings
from groq import Groq
import time

SYSTEM_PROMPT = """You are a precise, citation-first assistant. 
Use only the provided context. If unsure, say you don't know.
Cite sources inline like [1], [2] corresponding to the provided context chunks.
Keep answers concise and factual."""

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 600) -> str:
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (chat.choices[0].message.content or "").strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 600
    ) -> Dict[str, Any]:
        """Return text + usage + latency (seconds); handles dict or pydantic usage objects."""
        t0 = time.time()
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.time() - t0
        text = (chat.choices[0].message.content or "").strip()

        usage_raw = getattr(chat, "usage", None)

        # Normalize usage to a dict (works for dicts and Pydantic objects)
        def _get(attr, default=None):
            if usage_raw is None:
                return default
            if isinstance(usage_raw, dict):
                return usage_raw.get(attr, default)
            # Pydantic object or similar
            return getattr(usage_raw, attr, default)

        usage_norm = {
            "prompt_tokens": _get("prompt_tokens"),
            "completion_tokens": _get("completion_tokens"),
            "total_tokens": _get("total_tokens"),
        }

        return {
            "text": text,
            "latency_s": latency,
            "usage": usage_norm,
            "model": self.model,
        }