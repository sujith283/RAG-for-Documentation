from app.pipeline import RagPipeline
pipe = RagPipeline()
out = pipe.answer("What is the capital of France?")
print("ANSWER:", out["answer"])
print("SOURCES:", [(s["n"], s["source"]) for s in out["sources"]])
print("METRICS:", out["metrics"])
