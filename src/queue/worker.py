import redis
import time
import uuid  # for unique consumer name
from src.rag.ingestion import ingest_single_file

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

STREAM_NAME = "ingestion_queue"
GROUP_NAME = "workers"
CONSUMER_NAME = f"worker_{uuid.uuid4().hex[:8]}"  # Unique per run

# create group if not exists
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
            # Optional: move to dead-letter stream
            r.xadd("dead_letter", data)
            r.xack(STREAM_NAME, GROUP_NAME, msg_id)