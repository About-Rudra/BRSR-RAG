import os
import pdfplumber
import uuid
import warnings
from qdrant_client.http.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue, FilterSelector, PointStruct,
    PayloadSchemaType,
)
from src.rag.embeddings import embed
from src.rag.vector_client import get_client, COLLECTION


def ensure_collection():
    """Create the collection on first use (not at import, so importing never needs a live DB)."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
        )
        # Filtered search on category/source needs payload indexes on Qdrant Cloud.
        # (Embedded mode ignores them and warns; the warning is harmless.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for field in ("category", "source"):
                client.create_payload_index(COLLECTION, field_name=field,
                                            field_schema=PayloadSchemaType.KEYWORD)


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def point_id(category, source, i):
    """Deterministic ID: the same chunk of the same file always gets the same ID,
    so re-ingesting overwrites instead of duplicating. (Python's hash() is
    randomised per run and can be negative, which Qdrant rejects.)"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{category}/{source}:{i}"))


def ingest_single_file(file_path, category):
    """
    Ingest ONE file into Qdrant.
    Called by queue workers.
    """

    print(f"[ingest] Ingesting file: {file_path}")

    if not os.path.exists(file_path):
        print("[ingest] File does not exist, skipping")
        return

    # Read page by page so every chunk knows which page it came from.
    pages = []  # (page_number, text), 1-based
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages.append((page_number, page_text))
    except Exception as e:
        print("[ingest] PDF read failed:", e)
        return

    if sum(len(t.strip()) for _, t in pages) < 200:
        print("[ingest] Skipped (too little text)")
        return

    # Chunk each page separately: a chunk never spans two pages.
    chunks = []  # (page_number, chunk_text)
    for page_number, page_text in pages:
        for chunk in chunk_text(page_text):
            chunks.append((page_number, chunk))

    if not chunks:
        print("[ingest] Skipped (no chunks)")
        return

    ensure_collection()  # fail fast if Qdrant is unreachable, before the slow embedding step

    vectors = embed([c for _, c in chunks])
    source = os.path.basename(file_path)

    points = []
    for i, ((page_number, chunk), vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=point_id(category, source, i),
            vector=vector.tolist(),
            payload={
                "text": chunk,
                "source": source,
                "category": category,
                "page": page_number
            }
        ))

    # Remove this file's old chunks first, so a re-ingested file that now
    # produces fewer chunks does not leave stale ones behind.
    client = get_client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=source)),
            FieldCondition(key="category", match=MatchValue(value=category)),
        ])),
    )

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    print(f"[ingest] Stored {len(points)} chunks from {file_path}")
