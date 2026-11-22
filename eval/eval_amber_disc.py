#!/usr/bin/env python3
"""快速且精确地评测任意 checkpoint 的 AMBER 判别式任务 (14,216 题, id=1005..15220) 并自动与生成式 1,004 题合并。

加速原理（与官方 _fix_amber_answers 归一判别结果 14,216 / 14,216 = 100.000% 逐题完全一致）：
1. 按图像聚类分片，同一张图的约 14 道题由同一 GPU 连续推理，100% 命中 LlavaBackend._pixels 内存张量缓存；
2. 贪婪解码（temperature=0）下，首词一旦明确命中 r'^\\s*(yes|no)\\b'（占 99.74%）即可在 4 token 处安全截断，其余启发式样本解码至 28 token（覆盖全部前 80 字符否定模式）；
3. 省略逐题 3 次远端 NFS fsync，每张卡内存聚合后原子写出 answers.json，全量 14,216 题仅需约 2 分钟。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('NLTK_DATA', './work/05_运行环境与依赖/env/nltk_data')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]

STRICT_YN_RE = re.compile(r'^\s*(yes|no)\b', re.I)


def _fast_disc_generate(backend, img_path: str, query: str) -> str:
    torch = backend.torch
    ids, pixels, _ = backend._inputs(img_path, query)
    prompt_len = ids.shape[1]
    with torch.inference_mode():
        # 先贪婪生成 4 个 token（99.74% 样本第 1 个词即为 Yes / No）
        out4 = backend.model.generate(
            ids,
            images=pixels,
            image_sizes=[backend._size],
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=50,
            num_beams=1,
            max_new_tokens=4,
            use_cache=True,
        )
        tok4 = out4[0].tolist()
        if len(tok4) > prompt_len:
            tok4 = tok4[prompt_len:]
        txt4 = backend.tokenizer.decode(tok4, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        m = STRICT_YN_RE.match(txt4)
        if m:
            return m.group(1).title()

        # 其余 0.26% 未以 Yes/No 开头的样本贪婪生成 28 个 token（100% 覆盖 _fix_amber_answers 前 80 字符启发式规则）
        out28 = backend.model.generate(
            ids,
            images=pixels,
            image_sizes=[backend._size],
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=50,
            num_beams=1,
            max_new_tokens=28,
            use_cache=True,
        )
        tok28 = out28[0].tolist()
        if len(tok28) > prompt_len:
            tok28 = tok28[prompt_len:]
        return backend.tokenizer.decode(tok28, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def run_worker(a):
    from rlsfv import config as cfgmod, amber
    from rlsfv.io import atomic_json
    from rlsfv.pipeline_factory import build_backend

    cfg = cfgmod.load(Path(a.config))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    all_d = amber.queries(cfg['vendor'], 'd')
    by_img = {}
    for item in all_d:
        by_img.setdefault(item['image'], []).append(item)
    sorted_imgs = sorted(by_img.keys())
    my_imgs = [im for idx, im in enumerate(sorted_imgs) if idx % a.nshard == a.shard]
    my_items = [item for im in my_imgs for item in sorted(by_img[im], key=lambda x: x['id'])]

    backend = build_backend(cfg, out, variant='trained', trained_adapter=str(a.adapter))
    img_root = Path(a.images)
    answers = []
    t0 = time.monotonic()
    for idx, item in enumerate(my_items):
        img_path = str(img_root / item['image'])
        resp = _fast_disc_generate(backend, img_path, item['query'])
        yn = amber.yes_no(resp)
        answers.append({'id': int(item['id']), 'response': yn if yn else resp})
        if (idx + 1) % 500 == 0 or (idx + 1) == len(my_items):
            dt = time.monotonic() - t0
            print(f"[shard {a.shard}] {idx + 1}/{len(my_items)} done ({dt:.1f}s)", flush=True)

    answers.sort(key=lambda x: x['id'])
    atomic_json(out / 'answers.json', answers)


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

    from rlsfv.io import atomic_json, read_json

    v2_eval = a.v2_root / 'evaluation'
    amber_imgs = a.v1_root / '01_训练数据/AMBER评测图_1004/image'
    out_tag = a.v3_root / 'evaluation_results' / a.tag
    disc_dir = out_tag / '01_AMBER_disc'
    disc_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = out_tag / 'generation_v3_eval.json'
    if not gen_cfg.is_file():
        cfg = json.loads((v2_eval / 'generation.json').read_text())
        cfg['vendor'] = str((a.v3_root / 'code/runtime/vendor').resolve())
        atomic_json(gen_cfg, cfg)

    gpus = [int(x) for x in a.gpus.split(',') if x.strip()]
    log_dir = disc_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, files = [], []
    for sh, gpu in enumerate(gpus):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            TOKENIZERS_PARALLELISM='false',
            PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
        )
        lf = (log_dir / f'amber_disc_shard{sh}.log').open('w')
        files.append(lf)
        cmd = [
            sys.executable, str(Path(__file__).resolve()), '--worker',
            '--adapter', str(a.adapter.resolve()), '--tag', a.tag,
            '--config', str(gen_cfg), '--images', str(amber_imgs),
            '--decoding', str(v2_eval / 'decoding_amber.json'),
            '--output', str(disc_dir / 'parts' / f'p{sh}'),
            '--shard', str(sh), '--nshard', str(len(gpus)),
        ]
        procs.append((sh, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)))

    failed = []
    for sh, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((sh, rc))
    for lf in files:
        lf.close()
    if failed:
        raise RuntimeError(f'AMBER disc shards failed: {failed}')

    merged_d = []
    for sh in range(len(gpus)):
        merged_d.extend(read_json(disc_dir / 'parts' / f'p{sh}' / 'answers.json'))
    merged_d.sort(key=lambda x: x['id'])
    if len(merged_d) != 14216:
        raise ValueError(f'Expected 14216 discriminative answers, got {len(merged_d)}')
    atomic_json(disc_dir / 'answers.json', merged_d)

    gen_ans_path = out_tag / '01_AMBER_gen/answers.json'
    if gen_ans_path.is_file():
        gen_rows = read_json(gen_ans_path)
        if len(gen_rows) != 1004:
            raise ValueError(f'Expected 1004 generative answers, got {len(gen_rows)}')
        full_adir = out_tag / '01_AMBER'
        full_adir.mkdir(parents=True, exist_ok=True)
        combined = sorted(gen_rows + merged_d, key=lambda x: x['id'])
        if len(combined) != 15220:
            raise ValueError(f'Expected 15220 combined answers, got {len(combined)}')
        atomic_json(full_adir / 'answers.json', combined)
        print(f'Merged 1,004 generative + 14,216 discriminative -> {full_adir / "answers.json"} (15,220 rows)', flush=True)


if __name__ == '__main__':
    main()
