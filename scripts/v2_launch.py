#!/usr/bin/env python3
"""启动并监控 4 个 GPU 分片工作者构造 E3-V3 偏好对（配额扫描限定于当次 run 目录）。"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]
from v2.storage import read, write
from scripts.run_training import idle_cards
from rlsfv.io import lock, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--skip-pilot-gate', action='store_true')
    a = p.parse_args()
    manifest = a.root / 'manifests' / ('pilot32.json' if a.pilot else 'source_index.json')
    subprocess.run([sys.executable, str(ROOT / 'scripts/v2_environment.py'), '--report', str(a.root / 'environment' / ('launch_' + a.run + '.json'))], check=True)
    if not a.pilot and not a.skip_pilot_gate:
        pilot = read(a.root / 'checks/pilot_acceptance.json')
        if not pilot.get('passed'):
            raise RuntimeError('Pilot acceptance has not passed')
        for name, h in pilot['production_code'].items():
            if sha256(ROOT / name) != h:
                raise ValueError('Code changed since accepted pilot: ' + name)
    idle_cards([0, 1, 2, 3])
    logs = a.root / 'logs' / a.run
    logs.mkdir(parents=True, exist_ok=True)
    run = a.root / 'data' / a.run
    run.mkdir(parents=True, exist_ok=True)
    with lock(run):
        processes = []
        handles = []
        for gpu in range(4):
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', TMPDIR='/opt/e3-build-tmp',
                HF_HOME=str(a.root / 'cache/huggingface'), TORCH_EXTENSIONS_DIR='/opt/e3-torch-extensions',
            )
            log = (logs / (str(gpu) + '.log')).open('a')
            handles.append(log)
            cmd = [sys.executable, str(ROOT / 'scripts/v2_worker.py'), '--root', str(a.root), '--manifest', str(manifest), '--run', a.run, '--shard', str(gpu)]
            processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        write(run / 'workers.json', [dict(gpu=i, pid=p.pid) for i, p in enumerate(processes)])
        try:
            last_size_scan = 0
            written_bytes = 0
            while True:
                codes = [p.poll() for p in processes]
                progress = []
                for i in range(4):
                    path = run / str(i) / 'progress.json'
                    if path.exists():
                        d = read(path)
                        progress.append({k: v for k, v in d.items() if k != 'images'})
                spaces = {str(pt): dict(free_bytes=os.statvfs(pt).f_bavail * os.statvfs(pt).f_frsize)
                          for pt in [a.root, Path('/'), Path('/localdisk-tmp')] if pt.exists()}
                if time.monotonic() - last_size_scan > 300:
                    # 仅统计当次 run 目录新增字节，避免误计历史目录与全盘扫描阻塞
                    written_bytes = sum(pt.lstat().st_size for pt in run.rglob('*') if pt.is_file() and not pt.is_symlink())
                    last_size_scan = time.monotonic()
                state = dict(status='building_pairs', updated_unix=time.time(), return_codes=codes, progress=progress,
                             storage=spaces, v3_run_bytes=written_bytes, training_started=False,
                             next_stage='A100 LoRA-DPO training after preflight acceptance')
                write(run / 'status.json', state)
                if any(c is not None and c != 0 for c in codes):
                    raise RuntimeError('Construction worker failed; see persistent shard logs')
                if any(s['free_bytes'] < 5 * 1024 ** 3 for s in spaces.values()):
                    raise RuntimeError('Less than 5 GiB free; stopping construction safely')
                if written_bytes > 20 * 1024 ** 3:
                    raise RuntimeError('V3 run output exceeds 20 GiB safety budget; inspect quota before continuing')
                if all(c == 0 for c in codes):
                    break
                time.sleep(20)
            subprocess.run([sys.executable, str(ROOT / 'scripts/v2_merge.py'), '--root', str(a.root), '--manifest', str(manifest), '--run', a.run], check=True)
            write(run / 'status.json', dict(status='pairs_complete_training_not_started', training_started=False, updated_unix=time.time(),
                                            next_stage='Run audit_v3_vs_v2.py then launch A100 training',
                                            summary=read(a.root / 'checks' / (a.run + '_summary.json'))))
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for handle in handles:
                handle.close()


if __name__ == '__main__':
    main()
