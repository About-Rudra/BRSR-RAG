import atexit
import os
import yaml
from qdrant_client import QdrantClient

with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

COLLECTION = config["collection_name"]

_client = None


def get_client():
    """Return the single shared QdrantClient for this process.

    QDRANT_MODE=local (default): embedded Qdrant stored in QDRANT_PATH (default vector_db/),
    no server needed. Embedded mode locks the folder, so only one client may exist per
    process -- that is why every module must go through this function.
    QDRANT_MODE=server: connect to QDRANT_HOST:QDRANT_PORT (e.g. docker).
    """
    global _client
    if _client is None:
        mode = os.getenv("QDRANT_MODE", "local").lower()
        if mode == "server":
            _client = QdrantClient(
                host=os.getenv("QDRANT_HOST", config.get("qdrant_host", "localhost")),
                port=int(os.getenv("QDRANT_PORT", config.get("qdrant_port", 6333))),
            )
        else:
            _client = QdrantClient(path=os.getenv("QDRANT_PATH", "vector_db"))
        # Close cleanly at exit (embedded mode otherwise prints an error during shutdown).
        atexit.register(_client.close)
    return _client
