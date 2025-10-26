from app.pipeline import RagPipeline
pipe = RagPipeline()
text = "France is a country in Europe. Paris is the capital of France. The Eiffel Tower is in Paris."
pipe.ingest_document(text, source="demo", title="France", section="facts")
print("Ingested.")
