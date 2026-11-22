from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "rlsfv.coverage-relation.v1"

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


_PATH_DEPENDENT_CONFIG_KEYS = frozenset({
    "_name_or_path", "name_or_path", "model_name_or_path", "tokenizer_name_or_path",
    "cache_dir", "device_map",
})


def stable_model_config_hash(config):
    "Content identity of a model config that ignores load-path fields."
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in _PATH_DEPENDENT_CONFIG_KEYS}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return fingerprint(clean(config))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def records(path):
    text = Path(path).read_text(encoding="utf8")
    value = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not isinstance(value, list):
        raise ValueError("Expected JSON array or JSONL")
    return value

@contextlib.contextmanager
def lock(directory):

    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another writer owns {directory}")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def code_hash():
    return fingerprint({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})

class Store:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.events = self.root / "events.jsonl"

    def event(self, data):
        with self.events.open("a", encoding="utf8") as f:
            f.write(json.dumps(dict(schema_version=SCHEMA, **data), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def path_for(self, request):
        return self.root / "cache" / (fingerprint([self.identity, request]) + ".json")

    def peek(self, request):
        """Cached result or None, without computing (used to batch only missing requests)."""
        path = self.path_for(request)
        if path.exists():
            self.event({"event": "cache_hit", "key": path.stem, "kind": request["kind"]})
            return read_json(path)["result"]
        return None

    def put(self, request, result):
        path = self.path_for(request)
        atomic_json(path, {"schema_version": SCHEMA, "request": request, "result": result})
        self.event({"event": "completed", "key": path.stem, "kind": request["kind"]})
        return result

    def cached(self, request, function):
        key = fingerprint([self.identity, request])
        path = self.root / "cache" / (key + ".json")
        if path.exists():
            result = read_json(path)["result"]
            self.event({"event": "cache_hit", "key": key, "kind": request["kind"]})
            return result
        self.event({"event": "request", "key": key, "request": request})
        try:
            result = function()
            atomic_json(path, {"schema_version": SCHEMA, "request": request, "result": result})
            self.event({"event": "completed", "key": key, "kind": request["kind"]})
            return result
        except Exception as exc:
            self.event({"event": "failed", "key": key, "error": repr(exc)})
            raise
