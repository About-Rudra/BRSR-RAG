from urllib import response
import yaml
import os
from qdrant_client import QdrantClient
from src.rag.embeddings import embed

with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)


COLLECTION = config["collection_name"]

def retrieve(query, top_k=6, score_threshold=0.2):
    query_vector = embed([query])[0].tolist()

    response = client.search(
    collection_name=COLLECTION,
    query_vector=query_vector,
    limit=top_k,
    with_payload=True,
)


    results = []
    for point in response:
        if point.score < score_threshold:
            continue

        payload = point.payload or {}
        text = payload.get("text")
        source = payload.get("source")

        if not text:
            continue

    results.append({
        "text": text,
        "source": source,
        "score": point.score
    })


    return results
