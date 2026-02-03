import os
import pdfplumber
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.rag.embeddings import embed

# Load config
with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

COLLECTION = config["collection_name"]

# Connect to Qdrant
client = QdrantClient(
    host=config["qdrant_host"],
    port=config["qdrant_port"]
)

# Ensure collection exists
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )

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


def ingest_single_file(file_path, category):
    """
    Ingest ONE file into Qdrant.
    Called by queue workers.
    """

    print(f"➡️ Ingesting file: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File does not exist, skipping")
        return

    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("❌ PDF read failed:", e)
        return

    if len(text.strip()) < 200:
        print("⚠️ Skipped (too little text)")
        return

    chunks = chunk_text(text)
    vectors = embed(chunks)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append({
            "id": hash(file_path + str(i)),
            "vector": vector.tolist(),
            "payload": {
                "text": chunk,
                "source": os.path.basename(file_path),
                "category": category
            }
        })

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    print(f"✅ Stored {len(points)} chunks from {file_path}")
