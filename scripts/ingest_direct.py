from app.retriever_pine import PineconeRetriever
from app.utils import sliding_window_chunk
from app.config import settings

text = "France is a country in Europe. Paris is the capital of France. The Eiffel Tower is in Paris."
chunks = sliding_window_chunk(text, settings.chunk_size_tokens, settings.chunk_overlap, meta={"source":"demo","title":"France","section":"facts"})
r = PineconeRetriever()
r.upsert_chunks(chunks)
print(f"Ingested {len(chunks)} chunks into namespace={settings.pinecone_namespace}.")
