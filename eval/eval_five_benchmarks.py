#!/usr/bin/env python3
"""E3-V3+ 五大基准并行全自动评测 + 无重叠同子集评估 + 多模型横向对比总表生成脚本。

评测范围（4×GPU 数据分片并行，零污染隔离输出）：
  01_AMBER               (1,004 生成式 + 14,216 判别式，含原始答案与官方 fix_answers 口径)
  02_POPE                (random / popular / adversarial 各 3,000 条 + 均值 + 2,970 无重叠同子集)
  03_CHAIR_Cover_COCO500 (COCO-500 全量 CHAIRs / CHAIRi / Cover / 物体数 / 词数 + 495 无重叠同子集)
  04_Object_HalBench     (300 图全量 Resp / Ment / 物体数 / 词数 + 299 无重叠同子集)
  05_ScienceQA           (4,241 题整体 / 图文子集 / 纯文本子集 / 三学科分项准确率)

对比表自动汇总模型：
  - 基座 LLaVA-1.5-7B
  - 原论文 RLSF-V (RLSF官方7B)
  - E3-V1
  - E3-V2
  - E3-V3+ (本次训练模型)

所有输出均写入 $V3_ROOT/evaluation_results/<tag>/，绝不修改 V1/V2 目录。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('NLTK_DATA', './work/05_运行环境与依赖/env/nltk_data')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]
from v2.storage import read, write


def _ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _log(log_path: Path, msg: str):
    line = f'[{_ts()}] {msg}'
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as f:
        f.write(line + '\n')


def _run_sharded(gpus, cmd_builder, log_dir: Path, prefix: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    files = []
    for shard, gpu in enumerate(gpus):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            TOKENIZERS_PARALLELISM='false',
            PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
        )
        lf = (log_dir / f'{prefix}_shard{shard}.log').open('w')
        files.append(lf)
        p = subprocess.Popen(cmd_builder(shard, len(gpus)), stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append((shard, p))
    failed = []
    for shard, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((shard, rc))
    for lf in files:
        lf.close()
    if failed:
        raise RuntimeError(f'{prefix} shards failed: {failed}; inspect {log_dir}')


def _fix_amber_answers(raw_path: Path, fixed_path: Path):
    ans = json.loads(raw_path.read_text())
    pat = re.compile(r'^\s*(yes|no)\b', re.I)
    out = []
    stats = {'caption': 0, 'strict_yn': 0, 'heuristic': 0}
    for x in ans:
        i, r = x['id'], x['response']
        if i <= 1004:
            out.append({'id': i, 'response': r})
            stats['caption'] += 1
            continue
        m = pat.match(r)
        if m:
            val = m[1].title()
            stats['strict_yn'] += 1
        else:
            low = r.lower()
            neg = (
                low.startswith('no') or 'there is no' in low[:80] or 'there are no' in low[:80]
                or ' not ' in low[:60] or 'without' in low[:60] or 'does not' in low[:60] or "isn't" in low[:60]
            )
            val = 'No' if neg else 'Yes'
            stats['heuristic'] += 1
        out.append({'id': i, 'response': val})
    fixed_path.write_text(json.dumps(out, ensure_ascii=False))
    return stats


def _merge_shard_rows(base: Path) -> list:
    rows = []
    for p in sorted(base.glob('parts/p*/rows.json')):
        rows += json.loads(p.read_text())
    rows.sort(key=lambda r: r['id'])
    write(base / 'rows.json', rows)
    return rows


def _write_eval_gen_config(v3_root: Path, v2_eval: Path, out_dir: Path) -> Path:
    cfg = json.loads((v2_eval / 'generation.json').read_text())
    cfg['vendor'] = str((v3_root / 'code/runtime/vendor').resolve())
    p = out_dir / 'generation_v3_eval.json'
    write(p, cfg)
    return p


def _pope_avg(p: dict, key: str = 'acc'):
    vals = [p[k][key] for k in ('random', 'popular', 'adversarial') if k in p and key in p[k]]
    return round(sum(vals) / len(vals), 2) if len(vals) == 3 else None


def _redline_audit(summary: dict) -> dict:
    a = summary.get('AMBER', {})
    c = summary.get('COCO500', summary.get('CHAIR_Cover', {}))
    o = summary.get('ObjectHalBench', summary.get('Object_HalBench', {}))
    p = summary.get('POPE', {})
    s = summary.get('ScienceQA', {})
    pope_avg = _pope_avg(p, 'acc') or 0.0
    checks = {}
    if a:
        checks['AMBER_CHAIR_le_RLSF_2.4'] = (a.get('generative.CHAIR', 999) <= 2.4, a.get('generative.CHAIR'))
        checks['AMBER_Hal_le_RLSF_5.1'] = (a.get('generative.Hal', 999) <= 5.1, a.get('generative.Hal'))
        checks['AMBER_Cover_ge_RLSF_plus7_40.3'] = (a.get('generative.Cover', 0) >= 40.3, a.get('generative.Cover'))
        checks['AMBER_Cog_le_0.3'] = (a.get('generative.Cog', 999) <= 0.3, a.get('generative.Cog'))
        checks['AMBER_RelF1_ge_68.0'] = (a.get('relation.F1', 0) >= 68.0, a.get('relation.F1'))
    if c:
        checks['COCO_CHAIRs_lt_RLSF_18.2'] = (c.get('chairs', 999) < 18.2, c.get('chairs'))
        checks['COCO_CHAIRi_lt_RLSF_10.2'] = (c.get('chairs_i', 999) < 10.2, c.get('chairs_i'))
        checks['COCO_Cover_gt_RLSF_42.78'] = (c.get('cover', 0) > 42.78, c.get('cover'))
    if o:
        checks['ObjHal_Resp_lt_RLSF_5.67'] = (o.get('resp', 999) < 5.67, o.get('resp'))
        checks['ObjHal_Ment_lt_RLSF_2.89'] = (o.get('ment', 999) < 2.89, o.get('ment'))
    if p:
        checks['POPE_AvgAcc_ge_85.91'] = (pope_avg >= 85.91, pope_avg)
    if s:
        checks['ScienceQA_Acc_ge_64.5'] = (s.get('acc', 0) >= 64.50, s.get('acc'))
    return {
        'pope_avg_acc': pope_avg,
        'checks': {k: {'pass': bool(ok), 'value': val} for k, (ok, val) in checks.items()},
        'all_passed': all(ok for ok, _ in checks.values()) if checks else False,
    }


def _compute_nonoverlap_subsets(v2_eval: Path, sdir: Path, out_root: Path) -> dict:
    """按 nonoverlap_test_indices.json 计算同子集无重叠附加指标（COCO 495, ObjHal 299, POPE 2970×3）。"""
    idx_file = v2_eval / 'nonoverlap_test_indices.json'
    if not idx_file.is_file():
        return {}
    indexes = read(idx_file)
    coco_ban = set(indexes.get('COCO500_excluded_image_ids', []))
    obj_ban = set(indexes.get('ObjectHalBench_excluded_row_ids', []))
    sub_root = out_root / 'nonoverlap_subsets'
    res = {}

    pdir = out_root / '02_POPE'
    if pdir.is_dir():
        res['POPE'] = {}
        for sp in ('random', 'popular', 'adversarial'):
            rf = pdir / sp / 'rows.json'
            if not rf.is_file():
                continue
            banned = set(indexes['POPE'][sp]['excluded_row_ids'])
            keep = [r for r in read(rf) if r['id'] not in banned]
            dest = sub_root / '02_POPE' / sp
            write(dest / 'parts/p0/rows.json', keep)
            subprocess.run([sys.executable, str(sdir / 'pope_stats.py'), '--dir', str(dest), '--out', str(dest / 'report.json')], check=True)
            res['POPE'][sp] = read(dest / 'report.json')

    cf = out_root / '03_CHAIR_Cover_COCO500/rows.json'
    if cf.is_file():
        keep = [r for r in read(cf) if r['id'] not in coco_ban]
        dest = sub_root / '03_CHAIR_Cover_COCO500'
        write(dest / 'parts/p0/rows.json', keep)
        subprocess.run([sys.executable, str(sdir / 'chair_stats.py'), str(v2_eval / 'data/annotations_trainval2014.zip'), str(dest)], check=True)
        res['COCO500'] = read(dest / 'report.json')

    of = out_root / '04_Object_HalBench/rows.json'
    if of.is_file():
        keep = [r for r in read(of) if r['id'] not in obj_ban]
        dest = sub_root / '04_Object_HalBench'
        write(dest / 'parts/p0/rows.json', keep)
        subprocess.run([
            sys.executable, str(sdir / 'objhal_stats.py'),
            str(v2_eval / 'data/annotations_trainval2014.zip'), str(dest), str(v2_eval / 'data/synonyms_refine.txt'),
        ], check=True)
        res['ObjectHalBench'] = read(dest / 'report.json')

    return res


def _load_historical_baselines(v1_eval: Path, v2_eval: Path) -> dict:
    """读取 Base、原论文 RLSF-V (RLSF官方7B)、E3-V1、E3-V2 的真实归档评测 JSON。"""
    models = {}
    v1_map = [
        ('Base_LLaVA_1_5_7B', 'base_LLaVA-1.5-7B'),
        ('RLSF_V_Official', 'RLSF官方7B'),
        ('E3_V1', 'E3_自反馈DPO训练模型'),
    ]
    for key, folder in v1_map:
        m = {}
        for cand_name in ('report_official_fixed.json', 'report_official.json', 'report.json'):
            af = v1_eval / '01_AMBER' / folder / cand_name
            if af.is_file():
                raw = read(af)
                m['AMBER'] = raw.get('metrics_percent', raw)
                break
        pope = {}
        for sp in ('random', 'popular', 'adversarial'):
            pf = v1_eval / '02_POPE' / folder / f'report_{sp}.json'
            if pf.is_file():
                pope[sp] = read(pf)
        if pope:
            m['POPE'] = pope
        cf = v1_eval / '03_CHAIR_Cover_COCO500' / folder / 'report.json'
        if cf.is_file():
            m['COCO500'] = read(cf)
        of = v1_eval / '04_Object_HalBench' / folder / 'report.json'
        if of.is_file():
            m['ObjectHalBench'] = read(of)
        sf = v1_eval / '05_ScienceQA' / folder / 'report.json'
        if sf.is_file():
            m['ScienceQA'] = read(sf)
        if m:
            models[key] = m

    v2_sum = v2_eval / 'results/E3_V2/five_benchmark_summary.json'
    if v2_sum.is_file():
        raw_v2 = read(v2_sum)
        models['E3_V2'] = {
            'AMBER': raw_v2.get('AMBER', {}),
            'POPE': raw_v2.get('POPE', {}),
            'COCO500': raw_v2.get('CHAIR_Cover', raw_v2.get('COCO500', {})),
            'ObjectHalBench': raw_v2.get('Object_HalBench', raw_v2.get('ObjectHalBench', {})),
            'ScienceQA': raw_v2.get('ScienceQA', {}),
        }
    return models


def _fmt(v, nd=2):
    if v is None or v == '—':
        return '—'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def _delta_str(new_v, ref_v, lower_is_better: bool, nd=2):
    if not isinstance(new_v, (int, float)) or not isinstance(ref_v, (int, float)):
        return '—'
    d = new_v - ref_v
    good = (d <= 0) if lower_is_better else (d >= 0)
    mark = '✅' if good else '⚠️'
    return f'{d:+.{nd}f} {mark}'


def _build_markdown_tables(tag: str, hist: dict, v3_summary: dict, v2_eval: Path, v3_nonoverlap: dict) -> str:
    b = hist.get('Base_LLaVA_1_5_7B', {})
    r = hist.get('RLSF_V_Official', {})
    v1 = hist.get('E3_V1', {})
    v2 = hist.get('E3_V2', {})
    v3 = {
        'AMBER': v3_summary.get('AMBER', {}),
        'POPE': v3_summary.get('POPE', {}),
        'COCO500': v3_summary.get('COCO500', {}),
        'ObjectHalBench': v3_summary.get('ObjectHalBench', {}),
        'ScienceQA': v3_summary.get('ScienceQA', {}),
    }

    def g(m, sec, k):
        return m.get(sec, {}).get(k)

    lines = [
        f'# E3-V3+ (`{tag}`) 五大基准全量评测与横向对比总表',
        '',
        '## 1. 核心五大基准全量对比（原论文 `RLSF-V` vs `E3-V2` vs **`E3-V3+`**，附 `Base` 与 `E3-V1`）',
        '',
        '| 评测基准 | 指标 | 基座 `LLaVA-1.5-7B` | **原论文 `RLSF-V`** | `E3-V1` | **`E3-V2`** | **`E3-V3+`（本次）** | Δ (`V3+` vs `RLSF-V`) | Δ (`V3+` vs `E3-V2`) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    rows_spec = [
        ('**AMBER 生成式**', '`CHAIR` ↓ (%)', 'AMBER', 'generative.CHAIR', True, 1),
        ('**AMBER 生成式**', '**`Cover` ↑ (%)**', 'AMBER', 'generative.Cover', False, 1),
        ('**AMBER 生成式**', '**`Hal` ↓ (%)**', 'AMBER', 'generative.Hal', True, 1),
        ('**AMBER 生成式**', '`Cog` ↓ (%)', 'AMBER', 'generative.Cog', True, 1),
        ('**AMBER 判别总体**', '`Accuracy` ↑ (%)', 'AMBER', 'discriminative.Accuracy', False, 1),
        ('**AMBER 判别总体**', '`Precision` ↑ (%)', 'AMBER', 'discriminative.Precision', False, 1),
        ('**AMBER 判别总体**', '`Recall` ↑ (%)', 'AMBER', 'discriminative.Recall', False, 1),
        ('**AMBER 判别总体**', '`F1` ↑ (%)', 'AMBER', 'discriminative.F1', False, 1),
        ('**AMBER 判别六类 F1**', '`Existence F1` ↑', 'AMBER', 'existence.F1', False, 1),
        ('**AMBER 判别六类 F1**', '`Attribute F1` ↑', 'AMBER', 'attribute.F1', False, 1),
        ('**AMBER 判别六类 F1**', '`State F1` ↑', 'AMBER', 'state.F1', False, 1),
        ('**AMBER 判别六类 F1**', '`Number F1` ↑', 'AMBER', 'number.F1', False, 1),
        ('**AMBER 判别六类 F1**', '`Action F1` ↑', 'AMBER', 'action.F1', False, 1),
        ('**AMBER 判别六类 F1**', '**`Relation F1` ↑**', 'AMBER', 'relation.F1', False, 1),
        ('**COCO-500**', '**`CHAIRs` ↓ (%)**', 'COCO500', 'chairs', True, 2),
        ('**COCO-500**', '`CHAIRi` ↓ (%)', 'COCO500', 'chairs_i', True, 2),
        ('**COCO-500**', '**`Cover` ↑ (%)**', 'COCO500', 'cover', False, 2),
        ('**COCO-500**', '`Objs/Cap`', 'COCO500', 'objs_per_caption', False, 2),
        ('**COCO-500**', '`Words/Cap`', 'COCO500', 'words_per_caption', False, 1),
        ('**Object HalBench**', '**`Resp` ↓ (%)**', 'ObjectHalBench', 'resp', True, 2),
        ('**Object HalBench**', '**`Ment` ↓ (%)**', 'ObjectHalBench', 'ment', True, 2),
        ('**Object HalBench**', '`Objs/Cap`', 'ObjectHalBench', 'objects_per_caption', False, 2),
        ('**Object HalBench**', '`Words/Cap`', 'ObjectHalBench', 'words_per_caption', False, 1),
    ]

    for bench_lbl, metric_lbl, sec, k, low_better, nd in rows_spec:
        vb, vr, vv1, vv2, vv3 = g(b, sec, k), g(r, sec, k), g(v1, sec, k), g(v2, sec, k), g(v3, sec, k)
        lines.append(
            f'| {bench_lbl} | {metric_lbl} | {_fmt(vb, nd)} | {_fmt(vr, nd)} | {_fmt(vv1, nd)} | {_fmt(vv2, nd)} | **{_fmt(vv3, nd)}** | {_delta_str(vv3, vr, low_better, nd)} | {_delta_str(vv3, vv2, low_better, nd)} |'
        )

    # POPE splits + avg
    for sp_lbl, sp_key in [('`Random Acc` ↑', 'random'), ('`Popular Acc` ↑', 'popular'), ('`Adversarial Acc` ↑', 'adversarial')]:
        vb = b.get('POPE', {}).get(sp_key, {}).get('acc')
        vr = r.get('POPE', {}).get(sp_key, {}).get('acc')
        vv1 = v1.get('POPE', {}).get(sp_key, {}).get('acc')
        vv2 = v2.get('POPE', {}).get(sp_key, {}).get('acc')
        vv3 = v3.get('POPE', {}).get(sp_key, {}).get('acc')
        lines.append(
            f'| **POPE** | {sp_lbl} | {_fmt(vb, 2)} | {_fmt(vr, 2)} | {_fmt(vv1, 2)} | {_fmt(vv2, 2)} | **{_fmt(vv3, 2)}** | {_delta_str(vv3, vr, False, 2)} | {_delta_str(vv3, vv2, False, 2)} |'
        )
    p_b, p_r, p_v1, p_v2, p_v3 = _pope_avg(b.get('POPE', {})), _pope_avg(r.get('POPE', {})), _pope_avg(v1.get('POPE', {})), _pope_avg(v2.get('POPE', {})), _pope_avg(v3.get('POPE', {}))
    lines.append(
        f'| **POPE** | **`三组平均 Acc` ↑** | {_fmt(p_b, 2)} | {_fmt(p_r, 2)} | {_fmt(p_v1, 2)} | {_fmt(p_v2, 2)} | **{_fmt(p_v3, 2)}** | {_delta_str(p_v3, p_r, False, 2)} | {_delta_str(p_v3, p_v2, False, 2)} |'
    )

    # ScienceQA overall + image + text_only
    for sq_lbl, sq_getter in [
        ('**`Overall Acc` ↑**', lambda m: m.get('ScienceQA', {}).get('acc')),
        ('`Image Subset Acc` ↑', lambda m: m.get('ScienceQA', {}).get('image', {}).get('acc')),
        ('`Text-Only Acc` ↑', lambda m: m.get('ScienceQA', {}).get('text_only', {}).get('acc')),
    ]:
        vb, vr, vv1, vv2, vv3 = sq_getter(b), sq_getter(r), sq_getter(v1), sq_getter(v2), sq_getter(v3)
        lines.append(
            f'| **ScienceQA** | {sq_lbl} | {_fmt(vb, 2)} | {_fmt(vr, 2)} | {_fmt(vv1, 2)} | {_fmt(vv2, 2)} | **{_fmt(vv3, 2)}** | {_delta_str(vv3, vr, False, 2)} | {_delta_str(vv3, vv2, False, 2)} |'
        )

    # Non-overlap common subset table if available
    ov_file = v2_eval / 'overlap_free_baselines/summary.json'
    if ov_file.is_file() and v3_nonoverlap:
        ov = read(ov_file).get('results', {})
        lines += [
            '',
            '## 2. 附加同子集无重叠检验表（`COCO-495` / `ObjHal-299` / `POPE-2970×3`）',
            '',
            '> 注：`E3-V3+` 训练池本身已通过同源补图实现 **10,000 张源图全量零重叠**；下表进一步在剔除历史 6 张重合图后的完全相同测试子集上做附加对齐对比。',
            '',
            '| 子集基准 | 指标 | 基座 `LLaVA-1.5-7B` | **原论文 `RLSF-V`** | `E3-V1` | **`E3-V3+`（本次）** |',
            '|---|---|---:|---:|---:|---:|',
        ]
        c_ov = ov.get('03_CHAIR_Cover_COCO500', {})
        o_ov = ov.get('04_Object_HalBench', {})
        p_ov = ov.get('02_POPE', {})
        for lbl, k, nd in [('`CHAIRs` ↓', 'chairs', 2), ('`CHAIRi` ↓', 'chairs_i', 2), ('`Cover` ↑', 'cover', 2), ('`Words/Cap`', 'words_per_caption', 1)]:
            lines.append(
                f'| **COCO-495** | {lbl} | {_fmt(c_ov.get("base_LLaVA-1.5-7B", {}).get("all", {}).get(k), nd)} | '
                f'{_fmt(c_ov.get("RLSF官方7B", {}).get("all", {}).get(k), nd)} | '
                f'{_fmt(c_ov.get("E3_自反馈DPO训练模型", {}).get("all", {}).get(k), nd)} | '
                f'**{_fmt(v3_nonoverlap.get("COCO500", {}).get(k), nd)}** |'
            )
        for lbl, k, nd in [('`Resp` ↓', 'resp', 2), ('`Ment` ↓', 'ment', 2), ('`Words/Cap`', 'words_per_caption', 1)]:
            lines.append(
                f'| **ObjHal-299** | {lbl} | {_fmt(o_ov.get("base_LLaVA-1.5-7B", {}).get("all", {}).get(k), nd)} | '
                f'{_fmt(o_ov.get("RLSF官方7B", {}).get("all", {}).get(k), nd)} | '
                f'{_fmt(o_ov.get("E3_自反馈DPO训练模型", {}).get("all", {}).get(k), nd)} | '
                f'**{_fmt(v3_nonoverlap.get("ObjectHalBench", {}).get(k), nd)}** |'
            )
        if 'POPE' in v3_nonoverlap:
            lines.append(
                f'| **POPE-2970×3** | `Avg Acc` ↑ | {_fmt(_pope_avg(p_ov.get("base_LLaVA-1.5-7B", {})), 2)} | '
                f'{_fmt(_pope_avg(p_ov.get("RLSF官方7B", {})), 2)} | '
                f'{_fmt(_pope_avg(p_ov.get("E3_自反馈DPO训练模型", {})), 2)} | '
                f'**{_fmt(_pope_avg(v3_nonoverlap.get("POPE", {})), 2)}** |'
            )

    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3-root', type=Path, default=Path('./work/vpsf_v'))
    ap.add_argument('--v2-root', type=Path, default=Path('./work/vpsf_v'))
    ap.add_argument('--v1-root', type=Path, default=Path('./work'))
    ap.add_argument('--adapter', type=Path, required=True, help='训练完成的模型目录或中间 checkpoint-{step} 目录')
    ap.add_argument('--tag', type=str, default='E3_V3', help='评测输出子目录名，例如 E3_V3 或 ckpt_600')
    ap.add_argument('--gpus', type=str, default='0,1,2,3')
    ap.add_argument('--benchmarks', type=str, default='amber,pope,chair,objhal,scienceqa',
                    help='逗号分隔的评测集子集，例如 chair,amber 或全部五大基准')
    a = ap.parse_args()

    adapter = a.adapter.resolve()
    if not (adapter / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError(f'Missing adapter_model.safetensors in {adapter}')
    if not (adapter / 'non_lora_trainables.bin').is_file():
        raise FileNotFoundError(f'Missing non_lora_trainables.bin in {adapter}')

    gpus = [int(x) for x in a.gpus.split(',') if x.strip()]
    wanted = {x.strip().lower() for x in a.benchmarks.split(',') if x.strip()}

    v2_eval = a.v2_root / 'evaluation'
    v1_eval = a.v1_root / '07_评测结果'
    sdir = v2_eval / 'scripts'
    amber_imgs = a.v1_root / '01_训练数据/AMBER评测图_1004/image'

    out_root = a.v3_root / 'evaluation_results' / a.tag
    out_root.mkdir(parents=True, exist_ok=True)
    run_log = out_root / 'run.log'
    gen_cfg = _write_eval_gen_config(a.v3_root, v2_eval, out_root)

    summary = {}
    _log(run_log, f'Starting V3+ evaluation tag={a.tag} adapter={adapter} benchmarks={sorted(wanted)} gpus={gpus}')

    if 'amber' in wanted:
        _log(run_log, '01_AMBER start')
        adir = out_root / '01_AMBER'
        adir.mkdir(parents=True, exist_ok=True)
        if not ((adir / 'answers.json').is_file() and len(read(adir / 'answers.json')) == 15220):
            _run_sharded(
                gpus,
                lambda sh, nsh: [
                    sys.executable, str(sdir / 'amber_shard.py'),
                    '--config', str(gen_cfg), '--images', str(amber_imgs),
                    '--variant', 'trained', '--adapter', str(adapter),
                    '--output', str(adir / 'parts' / f'p{sh}'),
                    '--decoding', str(v2_eval / 'decoding_amber.json'),
                    '--shard', str(sh), '--nshard', str(nsh),
                ],
                adir / 'logs', 'amber',
            )
            subprocess.run([sys.executable, str(sdir / 'amber_merge.py'), str(adir), str(len(gpus))], check=True)
        else:
            _log(run_log, '01_AMBER reusing existing complete answers.json (15220 rows)')

        # 同时保存未归一原始评估与官方 fix_answers 归一后评估（与 V2 完全一致）
        raw_out = adir / 'official_raw'
        if raw_out.exists() and not ((raw_out / 'report.json').is_file() and read(raw_out / 'report.json').get('status') == 'complete'):
            shutil.rmtree(raw_out)
        if not raw_out.exists():
            subprocess.run(
                [sys.executable, '-m', 'rlsfv', 'evaluate', '--config', str(gen_cfg),
                 '--answers', str(adir / 'answers.json'), '--output', str(raw_out), '--task', 'a'],
                cwd=str(a.v3_root / 'code/generation'), check=True,
            )
        raw_rep = read(raw_out / 'metrics.json') if (raw_out / 'metrics.json').exists() else read(raw_out / 'report.json')
        write(adir / 'report.json', raw_rep)

        fix_stats = _fix_amber_answers(adir / 'answers.json', adir / 'answers_fixed.json')
        write(adir / 'fix_stats.json', fix_stats)
        fix_out = adir / 'official_fixed'
        if fix_out.exists() and not ((fix_out / 'report.json').is_file() and read(fix_out / 'report.json').get('status') == 'complete'):
            shutil.rmtree(fix_out)
        if not fix_out.exists():
            subprocess.run(
                [sys.executable, '-m', 'rlsfv', 'evaluate', '--config', str(gen_cfg),
                 '--answers', str(adir / 'answers_fixed.json'), '--output', str(fix_out), '--task', 'a'],
                cwd=str(a.v3_root / 'code/generation'), check=True,
            )
        rep = read(fix_out / 'metrics.json') if (fix_out / 'metrics.json').exists() else read(fix_out / 'report.json')
        write(adir / 'report_official_fixed.json', rep)
        summary['AMBER'] = rep.get('metrics_percent', rep) if isinstance(rep, dict) else rep
        _log(run_log, '01_AMBER done')

    if 'pope' in wanted:
        pdir = out_root / '02_POPE'
        summary['POPE'] = {}
        for split in ('random', 'popular', 'adversarial'):
            _log(run_log, f'02_POPE {split} start')
            sp_dir = pdir / split
            if not (sp_dir / 'report.json').is_file():
                _run_sharded(
                    gpus,
                    lambda sh, nsh, sp=split, sd=sp_dir: [
                        sys.executable, str(sdir / 'pope_shard.py'),
                        '--config', str(gen_cfg),
                        '--parquet', str(v2_eval / 'data' / f'{sp}.parquet'),
                        '--split', sp, '--variant', 'trained', '--adapter', str(adapter),
                        '--output', str(sd / 'parts' / f'p{sh}'),
                        '--shard', str(sh), '--nshard', str(nsh),
                    ],
                    sp_dir / 'logs', f'pope_{split}',
                )
                subprocess.run([sys.executable, str(sdir / 'pope_stats.py'), '--dir', str(sp_dir), '--out', str(sp_dir / 'report.json')], check=True)
            rep_sp = read(sp_dir / 'report.json')
            write(pdir / f'report_{split}.json', rep_sp)
            if (sp_dir / 'rows.json').is_file():
                write(pdir / f'rows_{split}.json', read(sp_dir / 'rows.json'))
            summary['POPE'][split] = rep_sp
            _log(run_log, f'02_POPE {split} done')

    if 'chair' in wanted:
        _log(run_log, '03_CHAIR_Cover_COCO500 start')
        cdir = out_root / '03_CHAIR_Cover_COCO500'
        if not (cdir / 'report.json').is_file():
            _run_sharded(
                gpus,
                lambda sh, nsh: [
                    sys.executable, str(sdir / 'chair_shard.py'),
                    '--config', str(gen_cfg), '--images', str(v2_eval / 'coco500'),
                    '--variant', 'trained', '--adapter', str(adapter),
                    '--output', str(cdir / 'parts' / f'p{sh}'),
                    '--shard', str(sh), '--nshard', str(nsh),
                ],
                cdir / 'logs', 'chair',
            )
            _merge_shard_rows(cdir)
            subprocess.run([sys.executable, str(sdir / 'chair_stats.py'), str(v2_eval / 'data/annotations_trainval2014.zip'), str(cdir)], check=True)
        summary['COCO500'] = read(cdir / 'report.json')
        summary['CHAIR_Cover'] = summary['COCO500']
        _log(run_log, '03_CHAIR_Cover_COCO500 done')

    if 'objhal' in wanted:
        _log(run_log, '04_Object_HalBench start')
        odir = out_root / '04_Object_HalBench'
        if not (odir / 'report.json').is_file():
            _run_sharded(
                gpus,
                lambda sh, nsh: [
                    sys.executable, str(sdir / 'objhal_shard.py'),
                    '--config', str(gen_cfg), '--data', str(v2_eval / 'data/obj_halbench_300.jsonl'),
                    '--variant', 'trained', '--adapter', str(adapter),
                    '--output', str(odir / 'parts' / f'p{sh}'),
                    '--shard', str(sh), '--nshard', str(nsh),
                ],
                odir / 'logs', 'objhal',
            )
            _merge_shard_rows(odir)
            subprocess.run([
                sys.executable, str(sdir / 'objhal_stats.py'),
                str(v2_eval / 'data/annotations_trainval2014.zip'), str(odir), str(v2_eval / 'data/synonyms_refine.txt'),
            ], check=True)
        summary['ObjectHalBench'] = read(odir / 'report.json')
        summary['Object_HalBench'] = summary['ObjectHalBench']
        _log(run_log, '04_Object_HalBench done')

    if 'scienceqa' in wanted:
        _log(run_log, '05_ScienceQA start')
        qdir = out_root / '05_ScienceQA'
        if not (qdir / 'report.json').is_file():
            _run_sharded(
                gpus,
                lambda sh, nsh: [
                    sys.executable, str(sdir / 'scienceqa_shard.py'),
                    '--config', str(gen_cfg),
                    '--questions', str(v2_eval / 'scienceqa/questions.json'),
                    '--sqa-dir', str(v2_eval / 'scienceqa'),
                    '--variant', 'trained', '--adapter', str(adapter),
                    '--output', str(qdir / 'parts' / f'p{sh}'),
                    '--decoding', str(v2_eval / 'decoding_scienceqa.json'),
                    '--shard', str(sh), '--nshard', str(nsh),
                ],
                qdir / 'logs', 'scienceqa',
            )
            subprocess.run([sys.executable, str(sdir / 'sqa_stats3.py'), '--dir', str(qdir), '--out', str(qdir / 'report.json')], check=True)
        summary['ScienceQA'] = read(qdir / 'report.json')
        _log(run_log, '05_ScienceQA done')

    # 计算无重叠公共子集指标并汇总历史对比表（Base / 原论文 RLSF-V / E3-V1 / E3-V2 / E3-V3+）
    nonoverlap = _compute_nonoverlap_subsets(v2_eval, sdir, out_root)
    if nonoverlap:
        write(out_root / 'nonoverlap_subset_summary.json', nonoverlap)
        summary['nonoverlap_subsets'] = nonoverlap

    summary['redline_audit'] = _redline_audit(summary)
    write(out_root / 'five_benchmark_summary.json', summary)

    hist = _load_historical_baselines(v1_eval, v2_eval)
    all_cmp = dict(hist)
    all_cmp[a.tag] = summary
    write(out_root / 'comparison_all_models.json', all_cmp)

    md_table = _build_markdown_tables(a.tag, hist, summary, v2_eval, nonoverlap)
    (out_root / 'comparison_tables_RLSF_V2_V3plus.md').write_text(md_table, encoding='utf-8')

    _log(run_log, f'ALL REQUESTED BENCHMARKS & COMPARISON TABLES COMPLETE -> {out_root / "five_benchmark_summary.json"}')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
