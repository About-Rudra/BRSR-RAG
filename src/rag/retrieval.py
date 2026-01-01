import yaml
from qdrant_client import QdrantClient
from src.rag.embeddings import embed

with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

client = QdrantClient(
    host=config["qdrant_host"],
    port=config["qdrant_port"]
)

COLLECTION = config["collection_name"]

def retrieve(query, top_k=6, score_threshold=0.2):
    query_vector = embed([query])[0].tolist()

    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[],
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    results = []
    for point in response.points:
        if point.score < score_threshold:
            continue

        results.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "score": point.score
        })

    return results
