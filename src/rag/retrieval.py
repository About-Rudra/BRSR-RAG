import os
import yaml
from qdrant_client import QdrantClient
from src.rag.embeddings import embed

# Load config
with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

QDRANT_HOST = os.getenv("QDRANT_HOST", config.get("qdrant_host", "localhost"))
QDRANT_PORT = int(os.getenv("QDRANT_PORT", config.get("qdrant_port", 6333)))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION = config["collection_name"]


def retrieve(query, top_k=6, score_threshold=0.2):
    """Return up to top_k chunks for the query, best first.

    Chunks scoring below score_threshold are dropped (pass None to keep all).
    Returns [] if the collection does not exist yet or nothing matches.
    """
    if not client.collection_exists(COLLECTION):
        return []

    query_vector = embed([query])[0].tolist()

    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        if score_threshold is not None and hit.score < score_threshold:
            continue

        payload = hit.payload or {}
        text = payload.get("text")
        if not text:
            continue

        results.append({
            "text": text,
            "source": payload.get("source", "unknown"),
            "category": payload.get("category"),
            "page": payload.get("page"),
            "score": hit.score,
        })

    return results
