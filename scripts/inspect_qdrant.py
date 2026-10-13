from src.rag.vector_client import get_client, COLLECTION

client = get_client()

collection = COLLECTION

# 1. Collection info
info = client.get_collection(collection)
print("\nCOLLECTION INFO")
print(info)

# 2. Scroll points (peek inside DB)
print("\nSAMPLE STORED POINTS")
points, _ = client.scroll(
    collection_name=collection,
    limit=3,
    with_payload=True,
    with_vectors=False
)

for p in points:
    print("\nID:", p.id)
    print("SOURCE:", p.payload.get("source"))
    print("CATEGORY:", p.payload.get("category"))
    print("TEXT PREVIEW:")
    print(p.payload.get("text")[:500])
