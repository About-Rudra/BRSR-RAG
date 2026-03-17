import os
import redis
import uuid
from src.rag.ingestion import ingest_single_file

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

STREAM_NAME = "ingestion_queue"
GROUP_NAME = "workers"
CONSUMER_NAME = f"worker_{uuid.uuid4().hex[:8]}"

# Create consumer group if not exists
try:
    r.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" not in str(e):
        raise

print(f"Worker started as {CONSUMER_NAME}, waiting for jobs...")

while True:
    messages = r.xreadgroup(
        GROUP_NAME,
        CONSUMER_NAME,
        {STREAM_NAME: ">"},
        count=1,
        block=5000
    )

    if not messages:
        continue

    stream, entries = messages[0]
    for msg_id, data in entries:
        file_path = data["file_path"]
        category = data["category"]

        try:
            ingest_single_file(file_path, category)
            r.xack(STREAM_NAME, GROUP_NAME, msg_id)
            print(f"[{CONSUMER_NAME}] Success: {file_path}")
        except Exception as e:
            print(f"[{CONSUMER_NAME}] Failed: {file_path} → {e}")
            r.xadd("dead_letter", data)
            r.xack(STREAM_NAME, GROUP_NAME, msg_id)
