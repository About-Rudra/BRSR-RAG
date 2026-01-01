from qdrant_client import QdrantClient
import yaml

with open('src/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

client = QdrantClient(host=config['qdrant_host'], port=config['qdrant_port'])
collection = config['collection_name']

count = client.count(collection_name=collection)
print(f"Total chunks: {count.count}")

if count.count == 0:
    print("Empty.")
else:
    points = client.scroll(collection_name=collection, limit=count.count)[0]
    sources = sorted(set(p.payload['source'] for p in points))
    print(f"\nSources: {sources}")

    print("\nFirst 5 chunks:")
    for p in points[:5]:
        print(f"\n--- From {p.payload['source']} ---")
        print(p.payload['text'][:600] + "..." if len(p.payload['text']) > 600 else p.payload['text'])