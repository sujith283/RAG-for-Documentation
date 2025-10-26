# tests/test_rerank_and_answer.py
from app.pipeline import RagPipeline

class DummyCohereRes:
    class R:
        def __init__(self, idx, score): self.index, self.relevance_score = idx, score
    def __init__(self): self.results = [self.R(0,0.99), self.R(1,0.95)]

def test_answer_monkeypatch(monkeypatch):
    # Dummy retriever with fixed hits
    class DummyRetriever:
        def embed(self, texts): return [[0.1]*384 for _ in texts]
        def retrieve(self, query, top_k):
            return [
                {"text":"Paris is the capital of France.","metadata":{"source":"doc1","position":0}},
                {"text":"The Eiffel Tower is in Paris.","metadata":{"source":"doc2","position":1}},
            ]
    # Dummy LLM
    class DummyLLM:
        def generate(self, messages, temperature=0.2, max_tokens=600):
            return "Paris is France's capital [1]."

    # Patch components
    monkeypatch.setattr("app.pipeline.PineconeRetriever", lambda: DummyRetriever())
    monkeypatch.setattr("app.pipeline.GroqLLM", lambda: DummyLLM())
    class DummyCohere:
        def rerank(self, model, query, documents, top_n):
            return DummyCohereRes()
    monkeypatch.setattr("app.pipeline.cohere.Client", lambda api_key: DummyCohere())

    pipe = RagPipeline()
    out = pipe.answer("What is France's capital?")
    assert "Paris" in out["answer"]
    assert any(s["n"] == 1 for s in out["sources"])
