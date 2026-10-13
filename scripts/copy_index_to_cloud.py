"""Copy the local embedded index (vector_db/) to a Qdrant Cloud cluster, without re-embedding.

Used to give the hosted Streamlit app the same index as your machine.

    QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333 QDRANT_API_KEY=... \
        python -m scripts.copy_index_to_cloud

Refuses to touch an existing non-empty collection unless --replace is given.
"""
import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from src.rag.vector_client import COLLECTION

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--replace", action="store_true", help="delete and recreate the cloud collection")
parser.add_argument("--local-path", default=os.getenv("QDRANT_PATH", "vector_db"))
args = parser.parse_args()

source = QdrantClient(path=args.local_path)
target = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)

total = source.count(COLLECTION).count
print(f"Local collection '{COLLECTION}': {total} points")

if target.collection_exists(COLLECTION):
    existing = target.count(COLLECTION).count
    if existing and not args.replace:
        raise SystemExit(f"Cloud collection already has {existing} points. Re-run with --replace to overwrite.")
    target.delete_collection(COLLECTION)

target.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
for field in ("category", "source"):
    target.create_payload_index(COLLECTION, field_name=field, field_schema=PayloadSchemaType.KEYWORD)

copied, offset = 0, None
while True:
    points, offset = source.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=True)
    if points:
        target.upsert(COLLECTION, points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        copied += len(points)
        print(f"  copied {copied}/{total}")
    if offset is None:
        break

print(f"Done. Cloud collection now has {target.count(COLLECTION).count} points.")
source.close()
