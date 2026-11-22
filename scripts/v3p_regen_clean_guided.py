#!/usr/bin/env python3
"""E3-V3P 增量式洁净引导重生成管线（select-bottlenecks | worker | launch | compare）。

核心机制（100% 盲标，零测试集词表或真值参与）：
1. 完整复用原始 V1 已保存的 Stage-0 (10 条 fresh) + Stage-1/2 (2 条 fresh) 候选、模糊熵、foil_calibration 及既有确定性双模板视觉核查账本；
2. 仅针对瓶颈图像增量重生成 Stage-1 (6 条 guided) 与 Stage-2 (4 条 guided) 共 10 条引导候选：
   - checklist.facts = 'verified', min_support = 1.0, min_singleton_p = 0.68（彻底阻断 24%+ 已反驳词进入关键词清单）；
   - prompt_policy = 'cumulative'（保留 s0+s1 累积双核查真物体记忆，消除 s2 遗忘 13.2 个 s0 真物体的断崖）；
   - prompt_style = 'v3p_clean', relation_module = True（正面引导 3-4 句 ~55-75 词高密度真实前景/背景物体与空间关系，源头抑制 or / 否定缺失 / 句尾空泛推测）；
3. 自动接续升级后 ledger_guard（仅保护双核查 ['supported', 'supported'] 实体）的 repair + naming_checks，输出标准 .json.gz 分片结果，并输出同切片 A/B 严格对比报告。
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from rlsfv.config import X0
from rlsfv.facts import caption_valid, ensure_existence, make_aligner, union
from rlsfv.io import sha256
from rlsfv.scoring import score
from rlsfv.statement_guard import apply_candidate as guard_statements
from v2.construction import construct, verify_reusing_exact
from v2.repair import POLICY
from v2.storage import ImageStore, read, write
from v2.v3p_candidate import OR_SENT_RE, NEG_SENT_RE, compile_strict_ledger, effective_dv_diversity_score, eval_span


def _fast_scan_chunk(args_tuple):
    chunk, base_dir_str = args_tuple
    base_dir = Path(base_dir_str)
    rows = []
    for sh, ent in chunk:
        iid = str(ent["id"])
        p = base_dir / str(sh) / "results" / (iid + ".json.gz")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            r = json.load(f)
        comp = compile_strict_ledger(r["facts"], r.get("naming_gates"))
        # 统计全图已知双核查归并具象头总数
        all_img_dv = {
            h for vtag, h, _, _ in comp[1] if vtag == 0
        }
        img_dv_div = effective_dv_diversity_score(all_img_dv)

        best_dv_div = 0.0
        best_dv_n = 0
        best_words = 0
        has_48 = False
        for rc in r["candidates"]:
            txt = rc["text"].strip()
            sents = [s.strip() for s in txt.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            # 快速无泄漏句扫描
            clean_sents = []
            for idx, raw_s in enumerate(sents):
                s = raw_s + "."
                if OR_SENT_RE.search(s) or NEG_SENT_RE.search(s):
                    if idx == 0:
                        clean_sents = []
                        break
                    continue
                dv, _, one, zero, ref, bcnt, boref = eval_span(s, comp)
                if one or zero or ref or bcnt > 0 or boref > 0:
                    if idx == 0:
                        clean_sents = []
                        break
                    continue
                clean_sents.append(s)
            if len(clean_sents) >= 2:
                joined = " ".join(clean_sents)
                w = len(joined.split())
                if w >= 48:
                    dv_j, _, one_j, zero_j, ref_j, bcnt_j, boref_j = eval_span(joined, comp)
                    if not (one_j or zero_j or ref_j or bcnt_j > 0 or boref_j > 0):
                        has_48 = True
                        d_div = effective_dv_diversity_score(dv_j)
                        if (d_div, len(dv_j)) > (best_dv_div, best_dv_n):
                            best_dv_div = d_div
                            best_dv_n = len(dv_j)
                            best_words = w

        # 瓶颈判定：无 >=48w 零泄漏候选，或零泄漏候选丢失 >=2.0 个图内已知双核查物体且多样性得分 < 8.0，或绝对双核查多样性 < 6.5
        lost_div = max(0.0, img_dv_div - best_dv_div)
        is_bottleneck = (
            (not has_48)
            or (best_dv_div < 8.0 and lost_div >= 2.0)
            or (best_dv_div < 6.5)
        )
        rows.append(
            dict(
                id=iid,
                shard=sh,
                has_48=has_48,
                best_dv_div=round(best_dv_div, 3),
                best_dv_n=best_dv_n,
                best_words=best_words,
                img_dv_div=round(img_dv_div, 3),
                lost_div=round(lost_div, 3),
                is_bottleneck=is_bottleneck,
            )
        )
    return rows


def cmd_select_bottlenecks(a):
    R = a.v3_root.resolve()
    entries = read(R / "manifests/source_index.json")
    base_dir = R / "data" / a.base_run
    indexed = [(sh, ent) for sh in range(4) for ent in entries[sh::4]]
    n_w = max(1, min(a.workers, 32))
    csize = math.ceil(len(indexed) / (n_w * 4))
    chunks = [indexed[i : i + csize] for i in range(0, len(indexed), csize)]

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        for part in ex.map(_fast_scan_chunk, [(ch, str(base_dir)) for ch in chunks]):
            all_rows.extend(part)

    bots = [r for r in all_rows if r["is_bottleneck"]]
    # 按挽救潜力排序：优先无 >=48w，其次图内已核查但因污染丢失最多的样本
    bots_sorted = sorted(bots, key=lambda r: (0 if not r["has_48"] else 1, -r["lost_div"], r["best_dv_div"], r["id"]))

    # 构造严格均衡的 200 张小范围上游实验切片（从 4 个原始分片各取前 50 张代表性瓶颈图）
    pilot_200 = []
    by_sh = collections.defaultdict(list)
    for r in bots_sorted:
        by_sh[r["shard"]].append(r)
    for sh in range(4):
        pilot_200.extend(by_sh[sh][:50])

    out_dir = R / "manifests/v3p_targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "pilot200_bottleneck.json", pilot_200)
    write(out_dir / "all_bottlenecks.json", bots_sorted)
    summary = {
        "total_images": len(all_rows),
        "no_clean_48w_count": sum(not r["has_48"] for r in all_rows),
        "bottleneck_count": len(bots_sorted),
        "pilot200_count": len(pilot_200),
        "regular_clean_count": len(all_rows) - len(bots_sorted),
        "regular_mean_dv_div": round(statistics.mean(r["best_dv_div"] for r in all_rows if not r["is_bottleneck"]), 3),
        "bottleneck_mean_dv_div_before": round(statistics.mean(r["best_dv_div"] for r in bots_sorted), 3),
        "pilot200_path": str(out_dir / "pilot200_bottleneck.json"),
        "all_bottlenecks_path": str(out_dir / "all_bottlenecks.json"),
    }
    write(out_dir / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _extract_guided_caption(pipeline, cid: str, stage: int, prompt: str, text: str, raw: dict) -> dict:
    res = pipeline.hybrid.extract(text)
    c = {
        "id": cid,
        "stage": stage,
        "branch": "guided",
        "text": text,
        "artifact_type": "caption",
        "prompt": prompt,
        "generation": raw,
        "facts": res["facts"],
        "pending_queries": res["pending_queries"],
        "extraction": {
            k: res[k] for k in ("audit", "pending_facts", "rejected_facts", "uncovered", "entities")
        },
    }
    c = guard_statements(c)
    c["facts"] = ensure_existence(c["facts"])
    if pipeline.cfg.get("extraction", {}).get("source_scope_guard", False):
        from rlsfv.source_scope_guard import apply_candidate as source_guard
        c = source_guard(c)
    return c


def cmd_worker(a):
    import spacy
    import torch
    from rlsfv.backend import LlavaBackend, PosParser
    from rlsfv.pipeline import Pipeline

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected 1 visible GPU per worker, got {torch.cuda.device_count()}")

    R = a.v3_root.resolve()
    base_dir = R / "data" / a.base_run
    out_run_dir = R / "data" / a.run
    worker_dir = out_run_dir / f"worker_{a.worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    for sh in range(4):
        (out_run_dir / str(sh) / "results").mkdir(parents=True, exist_ok=True)
        (out_run_dir / str(sh) / "sources_v3p").mkdir(parents=True, exist_ok=True)

    manifest_entries = read(R / "manifests/source_index.json")
    ent_by_id = {}
    for sh in range(4):
        for ent in manifest_entries[sh::4]:
            ent_by_id[str(ent["id"])] = (sh, ent)

    targets_raw = read(a.targets)
    my_targets = targets_raw[a.worker_id :: a.num_workers]

    backend = LlavaBackend("/llava-7b", worker_dir / "model", ROOT / "runtime/vendor", variant="base", context_len=4096)
    parser = PosParser("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    nlp_lg = spacy.load("en_core_web_lg")
    risk = read(ROOT / "models/results_v1_2.json")["risk_model"]
    rmodels = read(ROOT / "models/hc_reject_models.json")

    scratch_pipe = Path(f"/tmp/e3_v3p_pipe_{a.run}_w{a.worker_id}")
    scratch_pipe.mkdir(parents=True, exist_ok=True)
    cached_cfg_key = None
    pipeline = None

    started = time.monotonic()
    done_list = []
    fallbacks = []

    for idx, trow in enumerate(my_targets):
        iid = str(trow["id"] if isinstance(trow, dict) else trow)
        orig_sh, entry = ent_by_id[iid]
        dest = out_run_dir / str(orig_sh) / "results" / (iid + ".json.gz")
        if dest.exists():
            done_list.append(iid)
            continue

        base_res_path = base_dir / str(orig_sh) / "results" / (iid + ".json.gz")
        base_res = read(base_res_path)
        if base_res["image_sha256"] != entry["image_sha256"]:
            raise ValueError(f"Image hash mismatch on {iid}")

        try:
            src_path = Path(base_res["source_file"])
            src_bytes = src_path.read_bytes()
            if hashlib.sha256(src_bytes).hexdigest() != base_res["source_sha256"]:
                raise ValueError(f"Source sha256 mismatch on {iid}")
            source = json.loads(src_bytes.decode("utf-8"))

            cfg = deepcopy(source["generation_identity"]["config"])
            cfg["prompt_policy"] = "cumulative"
            cfg["batching"] = {"captions": True, "verification": True, "max_batch": 16}
            cfg["checklist"] = dict(
                cfg.get("checklist") or {},
                mode="compact",
                facts="verified",
                min_support=1.0,
                min_singleton_p=0.68,
                include_pending=False,
                prompt_style="v3p_clean",
                open_detail=True,
                open_observation="short",
                relation_module=True,
                relation_priority=True,
                overflow_policy="legacy_degrade",
                max_entities=28,
                max_details=32,
                max_detail_questions=48,
                clean_entities=True,
                refutation_priority=False,
            )
            cfg_key = json.dumps(cfg, sort_keys=True)
            if pipeline is None or cached_cfg_key != cfg_key:
                pipeline = Pipeline(cfg, backend, parser, scratch_pipe, nlp=nlp_lg)
                cached_cfg_key = cfg_key

            cache = ImageStore(
                out_run_dir / str(orig_sh) / "cache" / (iid + ".json.gz"),
                dict(run=a.run, worker=a.worker_id, image_sha256=entry["image_sha256"], base_source_sha256=base_res["source_sha256"]),
            )
            pipeline.store = cache
            if pipeline.hybrid is not None:
                pipeline.hybrid.store = cache
                pipeline.hybrid.adjudicator.store = cache
            pipeline.image_hash = entry["image_sha256"]
            pipeline.defer_subset_mass_review = True

            image = entry["image"]
            known_facts = {}
            for d in (source.get("facts") or {}, base_res.get("facts") or {}):
                known_facts.update(deepcopy(d))
            for g in (base_res.get("naming_gates") or {}).values():
                for pk, pf in (g.get("pending_checks") or {}).items():
                    known_facts.setdefault(pk, deepcopy(pf))

            s0_cands = deepcopy([c for c in source["candidates"] if c["stage"] == 0])
            s1_fresh = deepcopy(next(c for c in source["candidates"] if c["stage"] == 1 and c["branch"] == "fresh"))
            s2_fresh = deepcopy(next(c for c in source["candidates"] if c["stage"] == 2 and c["branch"] == "fresh"))
            if len(s0_cands) != 10:
                raise ValueError(f"Expected 10 s0 candidates on {iid}")

            img_q_dir = scratch_pipe / "q_meta" / iid
            img_q_dir.mkdir(parents=True, exist_ok=True)

            # Stage 0: 100% 缓存命中既有核查结果
            cum_s0 = union(s0_cands, pipeline.aligner)
            verify_reusing_exact(pipeline, image, cum_s0, known_facts)
            known_facts.update(cum_s0)

            # Stage 1: 洁净双核查关键词引导生成 6 条候选
            _, p1 = pipeline.guided_prompt(image, cum_s0, s0_cands, 1, img_q_dir)
            gen1 = pipeline.captions(image, source["record"], 1, [p1] * 6)
            s1_guided = [
                _extract_guided_caption(pipeline, f"{iid}:s1:c{slot}", 1, p1, txt, raw)
                for slot, (txt, raw) in enumerate(gen1)
            ]
            s1_cands = s1_guided + [s1_fresh]
            cum_s1 = union(s0_cands + s1_cands, pipeline.aligner)
            verify_reusing_exact(pipeline, image, cum_s1, known_facts)
            known_facts.update(cum_s1)

            # Stage 2: 累积 s0+s1 双核查记忆引导生成 4 条候选
            _, p2 = pipeline.guided_prompt(image, cum_s1, s0_cands + s1_cands, 2, img_q_dir)
            gen2 = pipeline.captions(image, source["record"], 2, [p2] * 4)
            s2_guided = [
                _extract_guided_caption(pipeline, f"{iid}:s2:c{slot}", 2, p2, txt, raw)
                for slot, (txt, raw) in enumerate(gen2)
            ]
            s2_cands = s2_guided + [s2_fresh]
            all_22 = s0_cands + s1_cands + s2_cands

            cum_s2 = union(all_22, pipeline.aligner)
            verify_reusing_exact(pipeline, image, cum_s2, known_facts)
            known_facts.update(cum_s2)

            # 仅为新增的 10 条 guided 候选计算公共上下文 X0 模糊熵
            for c in s1_guided + s2_guided:
                req = {
                    "kind": "common_context_fuzzy",
                    "image_sha256": entry["image_sha256"],
                    "prompt": X0,
                    "caption": c["text"],
                    "fuzzy": cfg["fuzzy"],
                    "sampling": cfg["generation"],
                    "history": [],
                }
                c["fuzzy"] = cache.cached(
                    req,
                    lambda c=c: backend.score(image, X0, c["text"], parser, cfg["fuzzy"], cfg["generation"]),
                )

            calibration = deepcopy(source.get("foil_calibration"))
            ranked_src = score(all_22, cum_s2, cfg["scoring"], cfg["seed"], calibration)
            upgraded_source = {
                "schema_version": source["schema_version"],
                "status": "complete",
                "record": deepcopy(source["record"]),
                "generation_identity": pipeline.identity,
                "image_sha256": entry["image_sha256"],
                "candidates": all_22,
                "facts": cum_s2,
                "snapshots": [],
                "ranking": ranked_src,
                "dependency_conflicts": [],
                "foil_calibration": calibration,
            }
            src_out = out_run_dir / str(orig_sh) / "sources_v3p" / (iid + ".json")
            write(src_out, upgraded_source)
            src_out_sha = sha256(src_out)

            # 同图复用未改动 fresh 候选修复特征与全图视觉核查种子
            v2_seed_art = dict(base_res, source_sha256=src_out_sha)
            entry_up = dict(entry, source=str(src_out), source_sha256=src_out_sha)
            res_final = construct(upgraded_source, entry_up, pipeline, nlp, risk, rmodels, v2_artifact=v2_seed_art)
            cache.flush()
            write(dest, res_final)
            done_list.append(iid)
        except Exception as exc:
            fallbacks.append({"id": iid, "error": repr(exc), "traceback": traceback.format_exc()})
            write(worker_dir / "fallbacks.json", fallbacks)
            write(dest, base_res)
            done_list.append(iid)

        if len(done_list) == 1 or len(done_list) % 5 == 0 or len(done_list) == len(my_targets):
            elapsed = time.monotonic() - started
            rate = len(done_list) / max(elapsed, 1e-6)
            prog = {
                "worker_id": a.worker_id,
                "status": "complete" if len(done_list) == len(my_targets) else "running",
                "done": len(done_list),
                "total": len(my_targets),
                "fallbacks": len(fallbacks),
                "elapsed_sec": round(elapsed, 1),
                "sec_per_img": round(1.0 / max(rate, 1e-6), 3),
                "updated_unix": time.time(),
            }
            write(worker_dir / "progress.json", prog)
            print(json.dumps(prog, ensure_ascii=False), flush=True)


def evaluate_ab_on_targets(v3_root: Path, base_run: str, new_run: str, targets_path: Path) -> dict:
    """在完全相同的目标图像集上对比 base_run 与 new_run（及单图逐点 Pareto 合并）的零泄漏覆盖指标。"""
    targets = read(targets_path)
    manifest_entries = read(v3_root / "manifests/source_index.json")
    sh_map = {str(ent["id"]): sh for sh in range(4) for ent in manifest_entries[sh::4]}

    def _best_clean_metrics(r: dict):
        comp = compile_strict_ledger(r["facts"], r.get("naming_gates"))
        best_div, best_n, best_w, has_48 = 0.0, 0, 0, False
        raw_or_guided = sum(
            1 for c in r["candidates"] if c["stage"] > 0 and c["branch"] == "guided" and OR_SENT_RE.search(c["text"])
        )
        for rc in r["candidates"]:
            txt = rc["text"].strip()
            sents = [s.strip() for s in txt.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            clean_s = []
            for idx, rs in enumerate(sents):
                s = rs + "."
                if OR_SENT_RE.search(s) or NEG_SENT_RE.search(s):
                    if idx == 0:
                        clean_s = []
                        break
                    continue
                _, _, one, zero, ref, bcnt, boref = eval_span(s, comp)
                if one or zero or ref or bcnt > 0 or boref > 0:
                    if idx == 0:
                        clean_s = []
                        break
                    continue
                clean_s.append(s)
            if len(clean_s) >= 2:
                joined = " ".join(clean_s)
                w = len(joined.split())
                if w >= 48:
                    dv_j, _, one_j, zero_j, ref_j, bcnt_j, boref_j = eval_span(joined, comp)
                    if not (one_j or zero_j or ref_j or bcnt_j > 0 or boref_j > 0):
                        has_48 = True
                        d_div = effective_dv_diversity_score(dv_j)
                        if (d_div, len(dv_j)) > (best_div, best_n):
                            best_div, best_n, best_w = d_div, len(dv_j), w
        return has_48, best_div, best_n, best_w, raw_or_guided

    b_stats, n_stats, p_stats = [], [], []
    improved = 0
    for trow in targets:
        iid = str(trow["id"] if isinstance(trow, dict) else trow)
        sh = sh_map[iid]
        rb = read(v3_root / "data" / base_run / str(sh) / "results" / (iid + ".json.gz"))
        rn = read(v3_root / "data" / new_run / str(sh) / "results" / (iid + ".json.gz"))
        mb = _best_clean_metrics(rb)
        mn = _best_clean_metrics(rn)
        mp = mn if (mn[0], mn[1], mn[2]) >= (mb[0], mb[1], mb[2]) else mb
        if (mn[0], mn[1], mn[2]) > (mb[0], mb[1], mb[2]):
            improved += 1
        b_stats.append(mb)
        n_stats.append(mn)
        p_stats.append(mp)

    def _agg(rows):
        n = max(1, len(rows))
        w_valid = [r[3] for r in rows if r[0]]
        return {
            "images": len(rows),
            "pct_has_clean_ge48w": round(100.0 * sum(r[0] for r in rows) / n, 2),
            "mean_dv_diversity_score": round(statistics.mean(r[1] for r in rows), 3),
            "mean_canon_dv_heads": round(statistics.mean(r[2] for r in rows), 3),
            "mean_clean_words_when_ge48w": round(statistics.mean(w_valid), 2) if w_valid else 0.0,
            "guided_residual_or_per_img": round(statistics.mean(r[4] for r in rows), 3),
        }

    return {
        "target_images": len(targets),
        "strictly_improved_images": improved,
        "pct_strictly_improved": round(100.0 * improved / max(1, len(targets)), 2),
        "base_clean_v3": _agg(b_stats),
        "regenerated_v3p_only": _agg(n_stats),
        "pointwise_pareto_merged": _agg(p_stats),
    }


def cmd_launch(a):
    R = a.v3_root.resolve()
    out_run_dir = R / "data" / a.run
    out_run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = R / "logs" / a.run
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    t0 = time.monotonic()
    for wid in range(a.num_workers):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(wid),
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TMPDIR="/opt/e3-build-tmp",
            TORCH_EXTENSIONS_DIR="/opt/e3-torch-extensions",
        )
        os.makedirs("/opt/e3-build-tmp", exist_ok=True)
        os.makedirs("/opt/e3-torch-extensions", exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--v3-root", str(R),
            "--base-run", a.base_run,
            "--run", a.run,
            "--targets", str(a.targets),
            "--worker-id", str(wid),
            "--num-workers", str(a.num_workers),
        ]
        lf = (log_dir / f"worker_{wid}.log").open("a", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        procs.append((wid, p, lf))

    while any(p.poll() is None for _, p, _ in procs):
        done_sum = 0
        tot_sum = 0
        fb_sum = 0
        for wid in range(a.num_workers):
            pp = out_run_dir / f"worker_{wid}" / "progress.json"
            if pp.exists():
                pr = read(pp)
                done_sum += pr.get("done", 0)
                tot_sum += pr.get("total", 0)
                fb_sum += pr.get("fallbacks", 0)
        print(
            json.dumps(
                {
                    "run": a.run,
                    "done": done_sum,
                    "total": tot_sum,
                    "fallbacks": fb_sum,
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                }
            ),
            flush=True,
        )
        time.sleep(12)

    for _, _, lf in procs:
        lf.close()

    bad = [(wid, p.returncode) for wid, p, _ in procs if p.returncode != 0]
    if bad:
        raise RuntimeError(f"Workers failed: {bad}; check {log_dir}")

    ab_report = evaluate_ab_on_targets(R, a.base_run, a.run, a.targets)
    ab_report["wall_clock_seconds"] = round(time.monotonic() - t0, 1)
    write(out_run_dir / "ab_comparison_report.json", ab_report)
    print(json.dumps(ab_report, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select-bottlenecks")
    p_sel.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    p_sel.add_argument("--base-run", type=str, default="full10000_clean_v3")
    p_sel.add_argument("--workers", type=int, default=4)

    p_wrk = sub.add_parser("worker")
    p_wrk.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    p_wrk.add_argument("--base-run", type=str, default="full10000_clean_v3")
    p_wrk.add_argument("--run", type=str, required=True)
    p_wrk.add_argument("--targets", type=Path, required=True)
    p_wrk.add_argument("--worker-id", type=int, required=True)
    p_wrk.add_argument("--num-workers", type=int, default=4)

    p_lch = sub.add_parser("launch")
    p_lch.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    p_lch.add_argument("--base-run", type=str, default="full10000_clean_v3")
    p_lch.add_argument("--run", type=str, required=True)
    p_lch.add_argument("--targets", type=Path, required=True)
    p_lch.add_argument("--num-workers", type=int, default=4)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    p_cmp.add_argument("--base-run", type=str, default="full10000_clean_v3")
    p_cmp.add_argument("--run", type=str, required=True)
    p_cmp.add_argument("--targets", type=Path, required=True)

    a = ap.parse_args()
    if a.cmd == "select-bottlenecks":
        cmd_select_bottlenecks(a)
    elif a.cmd == "worker":
        cmd_worker(a)
    elif a.cmd == "launch":
        cmd_launch(a)
    elif a.cmd == "compare":
        rep = evaluate_ab_on_targets(a.v3_root.resolve(), a.base_run, a.run, a.targets)
        print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
