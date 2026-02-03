import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

STREAM_NAME = "ingestion_queue"

def enqueue_job(file_path, category, source):
    job = {
        "file_path": file_path,
        "category": category,
        "source": source
    }
    r.xadd(STREAM_NAME, job)
    print("📥 Enqueued:", job)
