import cohere
from app.config import settings

print("API key present:", bool(settings.cohere_api_key))
print("Model:", settings.cohere_model)

co = cohere.Client(settings.cohere_api_key)
docs = ["Paris is the capital of France.","Berlin is the capital of Germany.","Ottawa is the capital of Canada."]
res = co.rerank(model=settings.cohere_model, query="What is France's capital?", documents=docs, top_n=2)
top = res.results[0]
print("Top idx:", top.index, "Score:", top.relevance_score)
print("Top text:", docs[top.index])
