# tests/test_retriever_skeleton.py
import types
from app.retriever_pine import PineconeRetriever

class DummyIndex:
    def upsert(self, vectors, namespace): return {"upserted_count": len(vectors)}
    def query(self, vector, top_k, include_metadata, namespace):
        return {"matches":[{"id":"a:0","score":0.9,"metadata":{"text":"alpha","source":"A","position":0}},
                           {"id":"b:0","score":0.8,"metadata":{"text":"beta","source":"B","position":0}}]}
class DummyPC:
    def __init__(self,*a,**k): pass
    def list_indexes(self): return []
    def create_index(self, **k): return True
    def Index(self, name): return DummyIndex()

def test_retrieve_monkeypatch(monkeypatch):
    # Patch Pinecone client and embedder
    monkeypatch.setattr("app.retriever_pine.Pinecone", lambda api_key: DummyPC())
    class DummyEmbed:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.1]*384 for _ in texts]
    monkeypatch.setattr("app.retriever_pine.SentenceTransformer", lambda name: DummyEmbed())

    r = PineconeRetriever()
    hits = r.retrieve("hello", top_k=5)
    assert len(hits) == 2
    assert hits[0]["metadata"]["source"] == "A"
