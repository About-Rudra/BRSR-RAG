import os

# "sync" (default): ingest in-process right away.
# "redis": push a job onto the Redis stream; src/queue/worker.py consumes it.
INGEST_MODE = os.getenv("INGEST_MODE", "sync").lower()

STREAM_NAME = "ingestion_queue"

_redis = None


def _get_redis():
    # Imported lazily so redis is only needed when INGEST_MODE=redis.
    global _redis
    if _redis is None:
        import redis
        _redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _redis


def enqueue_job(file_path, category, source):
    """Add one ingestion job to the Redis stream."""
    _get_redis().xadd(STREAM_NAME, {
        "file_path": file_path,
        "category": category,
        "source": source,
    })
    print(f"Enqueued for ingestion: {file_path}")


def submit_ingest(file_path, category, source):
    """Ingest a file now (sync mode) or enqueue it (redis mode).

    Errors are logged, not raised, so one bad file does not stop a scrape.
    """
    try:
        if INGEST_MODE == "redis":
            enqueue_job(file_path, category, source)
        else:
            from src.rag.ingestion import ingest_single_file
            ingest_single_file(file_path, category)
    except Exception as e:
        print(f"Ingestion failed for {file_path}: {e}")
