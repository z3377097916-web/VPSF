#!/usr/bin/env python3
"""单 GPU 分片构造工作者（支持经图像与源哈希严格校验的只读 V2 缓存与同文特征复用）。"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]
from rlsfv.io import fingerprint, sha256, lock
from rlsfv.pipeline import Pipeline
from rlsfv.backend import LlavaBackend, PosParser
from v2.storage import read, write, ImageStore
from v2.construction import construct
from v2.repair import POLICY

DEFAULT_V2_DATA = Path('./work/vpsf_v/data/full10000_v2')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--v2-reuse-dir', type=Path, default=DEFAULT_V2_DATA)
    a = p.parse_args()
    entries = read(a.manifest)[a.shard::a.shards]
    out = a.root / 'data' / a.run / str(a.shard)
    out.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(ROOT)): sha256(p) for d in ['v2', 'generation/rlsfv', 'preference', 'scripts']
            for p in sorted((ROOT / d).glob('*.py'))}
    identity = dict(policy=POLICY, code=code, manifest_sha256=sha256(a.manifest), shard=a.shard, shards=a.shards)
    with lock(out):
        ip = out / 'identity.json'
        if ip.exists() and read(ip) != identity:
            raise ValueError('Run code or manifest changed; use another run name')
        write(ip, identity)
        import torch, spacy
        if torch.cuda.device_count() != 1:
            raise RuntimeError('A worker must see exactly one GPU')
        backend = LlavaBackend('/llava-7b', out / 'model', ROOT / 'runtime/vendor', variant='base', context_len=4096)
        write(out / 'model_metadata.json', backend.meta)
        parser = PosParser('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        nlp_lg = spacy.load('en_core_web_lg')
        risk = read(ROOT / 'models/results_v1_2.json')['risk_model']
        rmodels = read(ROOT / 'models/hc_reject_models.json')
        completed = []
        started = time.monotonic()
        last_progress_write = 0.0
        cached_cfg_key = None
        pipeline = None
        scratch_pipe = Path(f'/tmp/e3_v3_pipe_{a.run}_{a.shard}')
        scratch_pipe.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(entries):
            dest = out / 'results' / (entry['id'] + '.json.gz')
            if dest.exists():
                result = read(dest)
                if result['image_sha256'] != entry['image_sha256']:
                    raise ValueError('Completed image identity changed')
            else:
                try:
                    if sha256(entry['image']) != entry['image_sha256']:
                        raise ValueError('Image bytes changed after inventory')
                    if entry['requires_generation']:
                        cfg = read(ROOT / 'configs/generation.json')
                        generator = Pipeline(cfg, backend, parser, out / 'regenerated', nlp=nlp_lg)
                        source = generator.image(dict(id=entry['id'], image=entry['image'], split='train'))
                        if source['status'] != 'complete':
                            raise ValueError('Original pool regeneration failed: ' + str(source.get('error')))
                        raw = out / 'regenerated/images' / fingerprint(entry['id'])[:20] / 'result.json'
                        raw_sha = sha256(raw)
                    else:
                        raw = Path(entry['source'])
                        import hashlib
                        raw_bytes = None
                        for wait in (.1, .3, 1, 2, 4, 0):
                            try:
                                raw_bytes = raw.read_bytes()
                                break
                            except OSError:
                                if not wait:
                                    raise
                                time.sleep(wait)
                        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                        source = json.loads(raw_bytes.decode('utf-8'))
                    entry = dict(entry, source=str(raw), source_sha256=raw_sha)
                    cfg = json.loads(json.dumps(source['generation_identity']['config']))
                    cfg['batching']['max_batch'] = 8
                    cfg_key = json.dumps(cfg, sort_keys=True)
                    if pipeline is None or cached_cfg_key != cfg_key:
                        pipeline = Pipeline(cfg, backend, parser, scratch_pipe, nlp=nlp_lg)
                        cached_cfg_key = cfg_key

                    # 查找可选只读 V2 历史缓存与单图结果（优先当前同分片索引，严格校验图像哈希与源哈希）
                    fb_caches = []
                    v2_artifact = None
                    if a.v2_reuse_dir and a.v2_reuse_dir.exists():
                        shard_order = [a.shard] + [s for s in range(4) if s != a.shard]
                        for sh in shard_order:
                            cand_cache = a.v2_reuse_dir / str(sh) / 'cache' / (entry['id'] + '.json.gz')
                            if cand_cache.exists():
                                fb_caches.append(cand_cache)
                            cand_res = a.v2_reuse_dir / str(sh) / 'results' / (entry['id'] + '.json.gz')
                            if cand_res.exists():
                                loaded_v2 = read(cand_res)
                                if (loaded_v2.get('image_sha256') == entry['image_sha256']
                                        and loaded_v2.get('source_sha256') == entry['source_sha256']):
                                    v2_artifact = loaded_v2
                                break

                    cache = ImageStore(
                        out / 'cache' / (entry['id'] + '.json.gz'),
                        dict(run=identity, source_sha256=entry['source_sha256'], image_sha256=entry['image_sha256']),
                        fallback_paths=fb_caches,
                    )
                    pipeline.store = cache
                    if pipeline.hybrid is not None:
                        pipeline.hybrid.store = cache
                        pipeline.hybrid.adjudicator.store = cache
                    pipeline.image_hash = entry['image_sha256']
                    pipeline.defer_subset_mass_review = True
                    result = construct(source, entry, pipeline, nlp, risk, rmodels, v2_artifact=v2_artifact)
                    cache.flush()
                    write(dest, result)
                except Exception as exc:
                    write(out / 'failure.json', dict(id=entry['id'], error=repr(exc), traceback=traceback.format_exc()))
                    raise
            completed.append(dict(id=entry['id'], pairs=len(result['pairs']), edited_candidates=result['edited_candidates'],
                                  eligible_candidates=result['eligible_candidates'], skip_reason=result['skip_reason']))
            elapsed = time.monotonic() - started
            is_last = len(completed) == len(entries)
            now_mono = time.monotonic()
            if is_last or len(completed) == 1 or len(completed) % 5 == 0 or (now_mono - last_progress_write) >= 10.0:
                progress = dict(status='complete' if is_last else 'running', done=len(completed), total=len(entries),
                                elapsed_seconds=elapsed, recent_images=completed[-5:], updated_unix=time.time())
                write(out / 'progress.json', progress)
                last_progress_write = now_mono
                print(json.dumps({k: v for k, v in progress.items() if k != 'recent_images'}), entry['id'], 'pairs', len(result['pairs']), flush=True)
        write(out / 'complete.json', dict(images=len(completed), pairs=sum(x['pairs'] for x in completed), identity=identity))


if __name__ == '__main__':
    main()
