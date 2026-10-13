"""Human approval. Verified summaries wait in data/updates_pending.json; only an explicit
approve() moves one into data/updates_log.json. Nothing reaches the log any other way."""
import os
import uuid

from src.watch.registry import (load_json, save_json, now, load_registry, save_registry, set_status)

PENDING_PATH = os.path.join("data", "updates_pending.json")
LOG_PATH = os.path.join("data", "updates_log.json")


def load_pending():
    return load_json(PENDING_PATH, [])


def load_log():
    return load_json(LOG_PATH, [])


def add_pending(entry):
    pending = load_pending()
    entry = {"id": uuid.uuid4().hex[:8], "status": "pending_approval", "created": now(), **entry}
    pending.append(entry)
    save_json(PENDING_PATH, pending)
    return entry


def _decide(entry_id, status):
    pending = load_pending()
    entry = next((e for e in pending if e["id"] == entry_id and e["status"] == "pending_approval"), None)
    if entry is None:
        raise KeyError(f"No pending update with id {entry_id}")
    entry["status"] = status
    entry["decided"] = now()
    save_json(PENDING_PATH, pending)

    registry = load_registry()
    set_status(registry, entry["url"], status)
    save_registry(registry)
    return entry


def approve(entry_id):
    entry = _decide(entry_id, "approved")
    log = load_log()
    log.append(entry)
    save_json(LOG_PATH, log)
    return entry


def reject(entry_id):
    return _decide(entry_id, "rejected")
