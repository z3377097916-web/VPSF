"""持久化单图原子读写与请求缓存（支持挂载盘瞬态重试、脏位按需落盘与只读 V2 缓存回退复用）。"""
import gzip
import json
import os
import tempfile
import time
from pathlib import Path
from rlsfv.io import fingerprint


def read(path):
    p = Path(path)
    for attempt, delay in enumerate((0.1, 0.3, 1.0, 2.0, 4.0, 0.0)):
        try:
            with (gzip.open(p, 'rt') if str(p).endswith('.gz') else p.open('r')) as f:
                return json.load(f)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, EOFError):
            if attempt == 5:
                raise
            time.sleep(delay)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.atomic-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
            if str(path).endswith('.gz'):
                data = gzip.compress(data, compresslevel=3, mtime=0)
            raw.write(data)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ImageStore:
    """按精确请求指纹索引的单图压缩缓存，可选只读复用同图像同源哈希的历史验证缓存。"""
    def __init__(self, path, identity, fallback_paths=None):
        self.path = Path(path)
        self.identity = identity
        self.rows = {}
        self.fallback_rows = {}
        self.events = []
        self._dirty = not self.path.exists()
        if fallback_paths:
            for fp in fallback_paths:
                fp = Path(fp)
                if not fp.exists():
                    continue
                old = read(fp)
                old_id = old.get('identity') or {}
                # 仅当图像 SHA256 与原始生成记录 SHA256 完全一致时才允许复用完全相同请求的推理结果
                if (old_id.get('image_sha256') == identity.get('image_sha256')
                        and old_id.get('source_sha256') == identity.get('source_sha256')):
                    for k, v in (old.get('requests') or {}).items():
                        self.fallback_rows.setdefault(k, v)
        if self.path.exists():
            d = read(self.path)
            if d['identity'] != identity:
                raise ValueError('Cache identity changed; use a new run directory')
            self.rows = d['requests']
            self.events = d['events']

    def peek(self, request):
        fp = fingerprint(request)
        row = self.rows.get(fp)
        if row is not None:
            return row['result']
        fb = self.fallback_rows.get(fp)
        if fb is not None:
            # 命中只读历史缓存后同步写入本轮缓存账本，保持完整自包含审计
            self.rows[fp] = dict(request=request, result=fb['result'])
            self._dirty = True
            return fb['result']
        return None

    def put(self, request, result):
        self.rows[fingerprint(request)] = dict(request=request, result=result)
        self._dirty = True
        return result

    def cached(self, request, function):
        result = self.peek(request)
        return result if result is not None else self.put(request, function())

    def event(self, data):
        self.events.append(data)
        self._dirty = True

    def flush(self):
        if not self._dirty and self.path.exists():
            return
        write(self.path, dict(identity=self.identity, requests=self.rows, events=self.events))
        self._dirty = False
