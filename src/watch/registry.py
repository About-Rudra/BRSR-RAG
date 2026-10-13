"""Document registry (data/registry.json) and the scrape-and-diff step. Plain code, no LLM.

Each entry: {url, title, regime, sha256, first_seen, last_seen, status}
status: new | changed | irrelevant | ingested | pending_approval | approved | rejected
"""
import hashlib
import json
import os
from datetime import datetime, timezone

REGISTRY_PATH = os.path.join("data", "registry.json")
STATUSES = ("new", "changed", "irrelevant", "ingested", "pending_approval", "approved", "rejected")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def load_registry(path=REGISTRY_PATH):
    return load_json(path, {})  # url -> entry


def save_registry(registry, path=REGISTRY_PATH):
    save_json(path, registry)


def classify(registry, url, digest):
    """'new' if the URL was never seen, 'changed' if its file hash differs, else 'unchanged'."""
    entry = registry.get(url)
    if entry is None:
        return "new"
    if entry.get("sha256") != digest:
        return "changed"
    return "unchanged"


def record(registry, url, title, regime, digest, status):
    """Insert or update an entry, keeping first_seen."""
    assert status in STATUSES, status
    ts = now()
    entry = registry.get(url, {"url": url, "first_seen": ts})
    entry.update({"title": title, "regime": regime, "sha256": digest, "last_seen": ts, "status": status})
    registry[url] = entry
    return entry


def set_status(registry, url, status):
    assert status in STATUSES, status
    if url in registry:
        registry[url]["status"] = status
        registry[url]["last_seen"] = now()
