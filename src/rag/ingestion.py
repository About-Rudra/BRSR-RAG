import os
import pdfplumber
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.rag.embeddings import embed

# Load config
with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["raw_data_path"]
COLLECTION = config["collection_name"]

print("Connecting to Qdrant...")
client = QdrantClient(
    host=config["qdrant_host"],
    port=config["qdrant_port"]
)

# Create collection ONLY if missing
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )

def chunk_text(text, chunk_size=800, overlap=100):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= chunk_size:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks

def ingest_all():
    point_id = 0
    total_chunks = 0

    for root, _, files in os.walk(RAW_PATH):
        category = os.path.basename(root)

        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            path = os.path.join(root, file)
            print(f"\nProcessing [{category}] → {file}")

            text = ""
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print("PDF read failed:", e)
                continue

            if len(text.strip()) < 200:
                print("Skipped (too little text)")
                continue

            chunks = chunk_text(text)
            print(f"Chunks created: {len(chunks)}")

            vectors = embed(chunks)

            points = []
            for chunk, vector in zip(chunks, vectors):
                points.append({
                    "id": point_id,
                    "vector": vector.tolist(),
                    "payload": {
                        "text": chunk,
                        "source": file,
                        "category": category
                    }
                })
                point_id += 1

            client.upsert(
                collection_name=COLLECTION,
                points=points
            )

            total_chunks += len(chunks)

    print(f"\nIngestion complete → {total_chunks} chunks stored")

if __name__ == "__main__":
    ingest_all()
