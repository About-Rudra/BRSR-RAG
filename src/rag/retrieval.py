from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.rag.embeddings import embed
from src.rag.vector_client import get_client, COLLECTION


def retrieve(query, top_k=6, score_threshold=0.2, category=None):
    """Return up to top_k chunks for the query, best first.

    Chunks scoring below score_threshold are dropped (pass None to keep all).
    category restricts the search to one regime (sebi_brsr / cpcb_epr / ccts).
    Returns [] if the collection does not exist yet or nothing matches.
    """
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return []

    query_vector = embed([query])[0].tolist()

    query_filter = None
    if category:
        query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])

    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=query_filter,
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
