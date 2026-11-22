#!/usr/bin/env python3
"""快速并行评测任意 checkpoint 的 AMBER 生成式任务 (1,004 图, task='g')。"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('NLTK_DATA', './work/05_运行环境与依赖/env/nltk_data')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]


def run_worker(args):
    from rlsfv import config as cfgmod, amber
    from rlsfv.pipeline_factory import build_backend
    cfg = cfgmod.load(Path(args.config))
    orig = amber.queries

    def patched(vendor, task='g'):
        q = orig(vendor, task)
        return [x for i, x in enumerate(q) if i % args.nshard == args.shard]

    amber.queries = patched
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dec = json.loads(Path(args.decoding).read_text())
    backend = build_backend(cfg, out, variant='trained', trained_adapter=str(args.adapter))
    amber.infer(cfg, backend, Path(args.images), out, 'g', dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--v3-root', type=Path, default=Path('./work/vpsf_v'))
    ap.add_argument('--v2-root', type=Path, default=Path('./work/vpsf_v'))
    ap.add_argument('--v1-root', type=Path, default=Path('./work'))
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--gpus', type=str, default='0,1,2,3')
    ap.add_argument('--config', type=str)
    ap.add_argument('--images', type=str)
    ap.add_argument('--decoding', type=str)
    ap.add_argument('--output', type=str)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=4)
    a = ap.parse_args()

    if a.worker:
        run_worker(a)
        return

    gpus = [int(x) for x in a.gpus.split(',') if x.strip()]
    tag_root = a.v3_root / 'evaluation_results' / a.tag
    out_dir = tag_root / '01_AMBER_gen'
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = tag_root / 'generation_v3_eval.json'
    if not gen_cfg.is_file():
        cfg_obj = json.loads((a.v2_root / 'evaluation/generation.json').read_text())
        cfg_obj['vendor'] = str((a.v3_root / 'code/runtime/vendor').resolve())
        gen_cfg.write_text(json.dumps(cfg_obj, ensure_ascii=False, indent=2))
    imgs = a.v1_root / '01_训练数据/AMBER评测图_1004/image'
    dec = a.v2_root / 'evaluation/decoding_amber.json'

    procs = []
    files = []
    for sh, gpu in enumerate(gpus):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        lf_dir = out_dir / 'logs'
        lf_dir.mkdir(exist_ok=True)
        lf = (lf_dir / f'shard{sh}.log').open('w')
        files.append(lf)
        cmd = [
            sys.executable, str(Path(__file__).resolve()), '--worker',
            '--adapter', str(a.adapter.resolve()), '--tag', a.tag,
            '--config', str(gen_cfg), '--images', str(imgs), '--decoding', str(dec),
            '--output', str(out_dir / 'parts' / f'p{sh}'),
            '--shard', str(sh), '--nshard', str(len(gpus)),
        ]
        procs.append((sh, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)))
    for sh, p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f'shard {sh} failed rc={rc}')
    for lf in files:
        lf.close()

    ans = []
    for sh in range(len(gpus)):
        ans += json.loads((out_dir / 'parts' / f'p{sh}' / 'answers.json').read_text())
    ans.sort(key=lambda r: r['id'])
    assert len(ans) == 1004
    (out_dir / 'answers.json').write_text(json.dumps(ans, ensure_ascii=False))

    eval_out = out_dir / 'official_gen'
    if eval_out.exists():
        shutil.rmtree(eval_out)
    subprocess.run(
        [sys.executable, '-m', 'rlsfv', 'evaluate', '--config', str(gen_cfg),
         '--answers', str(out_dir / 'answers.json'), '--output', str(eval_out), '--task', 'g'],
        cwd=str(a.v3_root / 'code/generation'), check=True,
    )
    rep = json.loads((eval_out / 'report.json').read_text())
    print(json.dumps(rep.get('metrics_percent', rep), indent=2), flush=True)


if __name__ == '__main__':
    main()
