#!/usr/bin/env python3
"""从 10,000 张零重叠分片账本构建非截断在策略零泄漏偏好对 E3_V3_OPT.json。

核心设计（100% 盲标，严守四大安全禁区，零测试集真值或词表参与）：
1. 自然完整度非截断守卫（Natural Completeness Guard）：
   - 严禁删除句中句子（0% 非连续子集拼接）与严禁重度截断（强制 words >= 48, cut_words <= 18, cut_ratio <= 0.22，使得 cut >= 20w 恒为 0.0%），彻底根除 E3_V3Z / E3_V3P 中因句中截断导致的早停 EOS 偏置与 DPO KL 坍缩（保持 rewards/chosen 稳定在健康区间 ~ -4.5）。
2. 三级分层严格零泄漏准入（Hierarchical Zero-Leak Admission）：
   - 全层级硬性清零：0 视觉反驳（含全部 11 类命题及 pending_checks + 全图跨候选表面引用跨度）、0 未消解 'or'、0 显式否定缺失从句、0 双盲零支持具象实体（['unclear','unclear']）、0 反驳/零支持数量命题；
   - Tier 1（纯金牌优先，占有效配对图像 ~74%+）：0 单支持具象实体（n_1sup == 0）且 0 单支持数量（cnt_1sup == 0），一旦图内存在 >=1 条纯金牌候选即锁定纯金牌池，绝不混入银牌；
   - Tier 2/3（受控高置信回退，仅当全图无 Tier 1 时启用，并结合重生成洁净引导分片补救）：至多 1 个高置信单支持具象实体（min_1sup_p >= 0.62/0.64）。
3. 在策略 s0 优先与去重核查实体多样性选正 + 双异构强反驳负例配对：
   - 保持 ~71% Stage-0 自然采样正例与 ~10.5% Stage-0 负例（与 E3-V2 在策略分布严格对齐），同时叠加通用同义归并多样性奖励；
   - 当双路由选出同一最优纯金牌正例时，第二路由自动匹配图内次优独立多错长负例（C - R 校准于 ~ -29.2 词），兼顾每图精确等权归一化（spread == 0）与充足梯度步数（~928 步）。
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from rlsfv.io import sha256
from rlsfv.scoring import effective_statuses, verified
from scripts.extract_features import compact_record
from v2.ledger_guard import is_concrete_entity
from v2.repair import POLICY
from v2.storage import read, write
from v2.v3p_candidate import (
    compile_strict_ledger,
    effective_dv_diversity_score,
    eval_span,
    fact_p_stats,
    norm_head,
)
from preference.protocol import PROMPT
from preference.purity import MIX_COV_TOL, QV_COV_TOL, RULE, evaluate
from preference.selectors import (
    _concat_diff,
    anchor_score,
    linear,
    mix,
    normalized_weights,
    qv,
    reject_features,
    risk_features,
)

OR_RE = re.compile(r"\bor\b", re.IGNORECASE)
NEG_RE = re.compile(
    r"\b(?:with\s+no|without\s+(?:any\s+)?|no\s+visible|not\s+visible|cannot\s+be\s+seen|"
    r"void\s+of|free\s+of|neither\s+\w+\s+nor|devoid\s+of|absent|unclear)\b",
    re.IGNORECASE,
)
CBS_RE = re.compile(r"\bcan\s+be\s+seen\b", re.IGNORECASE)


def extra_feature_map(facts: dict, candidates: list[dict]) -> dict:
    """提取与冻结 50 维负例选择线性模型完全兼容的附加统计特征。"""
    out = {}
    for c in candidates:
        counts = {
            cat: {"supported": 0, "uncertain": 0, "unsupported": 0}
            for cat in ("object", "relation", "aux")
        }
        ds = []
        for k in c["claim_ids"]:
            f = facts[k]
            if f["modality"] != "definite" or f["polarity"] != "positive":
                continue
            cat = f["category"]
            ss = [x.get("status") for x in f.get("checks", [])]
            st = (
                "unsupported"
                if "refuted" in ss
                else ("supported" if ss and all(x == "supported" for x in ss) else "uncertain")
            )
            if cat in counts:
                counts[cat][st] += 1
            if cat == "object":
                q = (
                    0.0
                    if "refuted" in ss
                    else (
                        1.0
                        if ss.count("supported") == 2
                        else 0.5
                        if ss.count("supported") == 1 and ss.count("unclear") == 1
                        else 0.0
                    )
                )
                ds.append(1.0 - q)
        row = {}
        for cat, v in counts.items():
            row.update(
                {
                    f"{cat}_n": sum(v.values()),
                    f"{cat}_sup": v["supported"],
                    f"{cat}_unc": v["uncertain"],
                    f"{cat}_uns": v["unsupported"],
                }
            )
        row["Do"] = sum(ds) / len(ds) if ds else 0.0
        out[c["id"]] = row
    return out


def consensus_from_original(iid: str, cand_list: list[dict], facts: dict, orig_cands: list[dict]) -> tuple[dict, dict]:
    """保持冻结 13 维风险模型外部一致性特征语义完全不变。"""
    orig_elig = {c["id"] for c in orig_cands if not c["errors"]}

    def _objs(c):
        o = {}
        for k in c["claim_ids"]:
            f = facts[k]
            if f["modality"] != "definite" or f["polarity"] != "positive" or f["category"] != "object":
                continue
            name = " ".join(str(f.get("subject") or "").lower().split())
            ss = [x.get("status") for x in f.get("checks", [])]
            st = (
                "unsupported"
                if "refuted" in ss
                else ("supported" if ss and all(x == "supported" for x in ss) else "uncertain")
            )
            if name and (name not in o or (o[name] != "supported" and st == "supported")):
                o[name] = st
        return o

    originals = {c["id"]: _objs(c) for c in orig_cands if c["id"] in orig_elig}
    holders: dict[str, dict[str, str]] = {}
    for cid, ob in originals.items():
        for name, st in ob.items():
            holders.setdefault(name, {})[cid] = st
    cons, cons2 = {}, {}
    multi = len(originals) > 1
    for c in cand_list:
        ob = _objs(c)
        den = max(1, len(ob))
        both = sum(sum(v == "supported" for v in holders.get(name, {}).values()) >= 2 for name in ob) if multi else 0
        freq = sum(len(holders.get(name, {})) for name in ob) / den if multi else 0.0
        ext_c = no_ext = 0
        if multi:
            for name, st in ob.items():
                sup = any(cid != c["id"] and v == "supported" for cid, v in holders.get(name, {}).items())
                ext_c += int(st == "supported" and sup)
                no_ext += int(not sup)
        cons[(iid, c["id"])] = dict(obj_n=len(ob), cons_sup_frac=both / den, freq_mean=freq)
        cons2[(iid, c["id"])] = dict(ext_sup_cov_frac=ext_c / den, no_ext_frac=no_ext / den)
    return cons, cons2


def inspect_unchopped_candidate(
    c: dict,
    orig_w: int,
    facts: dict,
    gate: dict | None,
    compiled_ledger: tuple,
    min_words: int = 48,
    max_cut_words: int = 18,
    max_cut_ratio: float = 0.22,
) -> dict | None:
    """对候选执行非截断自然完整度校验 + 命题账本与全图表面跨度双重零泄漏审计。"""
    txt = c["text"]
    w = int(c["words"])
    cut = max(0, int(orig_w) - w)
    cut_ratio = (cut / orig_w) if orig_w > 0 else 0.0
    if w < min_words or cut > max_cut_words or cut_ratio > max_cut_ratio:
        return None
    if c["errors"] or (gate is not None and not gate.get("passed", False)):
        return None
    if OR_RE.search(txt) or NEG_RE.search(txt):
        return None

    p_checks = (gate or {}).get("pending_checks") or {}
    all_f = [facts.get(k) for k in c["claim_ids"]] + list(p_checks.values())
    dvh, one_h, zero_h = set(), set(), set()
    min_1sup_p = 1.0
    bad_cnt_0ref = 0
    cnt_1sup = 0

    for gf in all_f:
        if gf is None or not verified(gf):
            return None
        ss = effective_statuses(gf)
        if "refuted" in ss:
            return None
        pol, cat = gf.get("polarity"), gf.get("category")
        if pol == "negative" and cat in {"object", "identity"}:
            conc, _ = is_concrete_entity(gf)
            if conc:
                return None
        if pol == "positive" and cat in {"object", "identity"}:
            conc, h = is_concrete_entity(gf)
            if conc:
                nh = norm_head(h)
                if ss == ["supported", "supported"]:
                    dvh.add(nh)
                elif ss.count("supported") == 1:
                    one_h.add(nh)
                    _, mp = fact_p_stats(gf)
                    min_1sup_p = min(min_1sup_p, mp)
                else:
                    zero_h.add(nh)
        if pol == "positive" and cat == "count":
            if ss == ["supported", "supported"]:
                pass
            elif ss.count("supported") == 1:
                cnt_1sup += 1
            else:
                bad_cnt_0ref += 1

    one_h -= dvh
    zero_h -= (dvh | one_h)
    if zero_h or bad_cnt_0ref > 0:
        return None

    sdv, srel, s_one, s_zero, s_ref, s_bcnt, s_bref = eval_span(txt, compiled_ledger)
    s_zero -= (dvh | one_h)
    s_ref -= dvh
    if s_zero or s_ref or s_bref > 0:
        return None

    return dict(
        w=w,
        cut=cut,
        cut_ratio=round(cut_ratio, 4),
        n_1sup=len(one_h),
        min_1sup_p=round(min_1sup_p, 6),
        cnt_1sup=cnt_1sup,
        dv_div=round(effective_dv_diversity_score(dvh | sdv), 4),
        dv_rels=int(srel),
        is_s0=(":s0:" in c["id"]),
        has_cbs=bool(CBS_RE.search(txt)),
    )


def extract_tiered_pools(rec: dict) -> tuple[dict, dict, dict, list[dict], list[dict], list[dict], dict]:
    """从单图分片记录提取 Tier 1（纯金牌）、Tier 2（高置信银牌）、Tier 3（受控数量回退）候选池。"""
    facts = rec["facts"]
    gates = rec.get("naming_gates") or {}
    comp = compile_strict_ledger(facts, gates)
    old_im = rec["original_features"]
    new_im = rec.get("repaired_features")
    if new_im is None:
        new_im = compact_record(
            dict(
                status="complete",
                record=rec["record"],
                image_sha256=rec["image_sha256"],
                candidates=rec["candidates"],
                facts=facts,
                evidence_candidates=rec["evidence_candidates"],
                ranking=rec["ranking"],
            )
        )
    orig_by_id = {c["id"]: c for c in old_im["candidates"]}
    t1, t2, t3, info = [], [], [], {}
    for c in new_im["candidates"]:
        orig_w = orig_by_id[c["id"]]["words"]
        inf = inspect_unchopped_candidate(c, orig_w, facts, gates.get(c["id"]), comp)
        if inf is None:
            continue
        info[c["id"]] = inf
        if inf["n_1sup"] == 0 and inf["cnt_1sup"] == 0:
            t1.append(c)
        elif inf["n_1sup"] <= 1 and inf["min_1sup_p"] >= 0.62 and inf["cnt_1sup"] == 0:
            t2.append(c)
        elif inf["n_1sup"] <= 1 and inf["min_1sup_p"] >= 0.64 and inf["cnt_1sup"] <= 1:
            t3.append(c)
    return old_im, new_im, facts, t1, t2, t3, info


def select_opt_pairs_for_image(
    iid: str,
    img_path: str,
    img_sha: str,
    source_sha: str,
    source_contract: dict,
    old_im: dict,
    new_im: dict,
    facts: dict,
    pool: list[dict],
    info: dict,
    risk_model: dict,
    reject_models: dict,
) -> tuple[list[dict], list[dict]]:
    """在单图洁净候选池上运行在策略多样性双路由选正与双异构强反驳负例配对。"""
    orig_by_id = {c["id"]: c for c in old_im["candidates"]}
    cons, cons2 = consensus_from_original(iid, new_im["candidates"], facts, old_im["candidates"])
    fmap_old = extra_feature_map(facts, old_im["candidates"])
    fmap_new = extra_feature_map(facts, new_im["candidates"])
    allf_v2 = {iid + "|" + c["id"]: fmap_old[c["id"]] for c in old_im["candidates"]}

    def _risk(c):
        z = linear(risk_model, risk_features(c, cons[(iid, c["id"])], cons2[(iid, c["id"])]))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    risks = {c["id"]: _risk(c) for c in pool}
    base = max(pool, key=anchor_score)

    elig_a = [c for c in pool if risks[c["id"]] <= risks[base["id"]] + 0.022 and mix(c) >= mix(base) + 0.03]

    def _score_a(c):
        inf = info[c["id"]]
        s0_b = 0.020 if inf["is_s0"] else (-0.012 if inf["has_cbs"] else 0.0)
        gold_b = 0.018 if (inf["n_1sup"] == 0 and inf["cnt_1sup"] == 0) else 0.0
        return 0.76 * mix(c) + 0.24 * qv(c) + 0.008 * min(10.0, inf["dv_div"]) + s0_b + gold_b

    ch_a = max(elig_a, key=lambda c: (round(_score_a(c), 12), -risks[c["id"]])) if elig_a else base

    def _score_b(c):
        inf = info[c["id"]]
        ps = (0.78 * mix(c) + 0.22 * qv(c)) * (
            0.05 + 0.95 * (c["p"] or 0.0) * (1.0 - c["unsupported_primary_fraction"])
        ) ** 2
        mult = (1.035 if inf["is_s0"] else (0.95 if inf["has_cbs"] else 0.99)) * (
            1.03 if (inf["n_1sup"] == 0 and inf["cnt_1sup"] == 0) else 1.0
        )
        return ps * mult * (1.0 + 0.012 * min(10.0, inf["dv_div"]))

    ch_b = max(pool, key=lambda c: (round(_score_b(c), 12), -risks[c["id"]]))
    sel = {"v1.2": ch_a, "product": ch_b}

    used_rej_ids: set[str] = set()
    img_pairs = []
    c_audit = []

    for role, other in [("v1.2", "product"), ("product", "v1.2")]:
        ch = sel[role]
        mf = dict(allf_v2)
        mf[iid + "|" + ch["id"]] = fmap_new[ch["id"]]
        pc = float(ch["p"] or 0.0)
        mc = mix(ch)
        qvc = qv(ch)
        excl = {ch["id"], sel[other]["id"]} | (used_rej_ids if sel["v1.2"]["id"] == sel["product"]["id"] else set())

        def _valid_rej(r: dict, min_err: int, min_w_off: int) -> bool:
            if r["id"] in excl or len(r["errors"]) < min_err:
                return False
            if r["words"] < ch["words"] + min_w_off:
                return False
            if pc - float(r["p"] or 0.0) <= 1e-12:
                return False
            if (mc - mix(r)) < MIX_COV_TOL or (qvc - qv(r)) < QV_COV_TOL:
                return False
            return True

        rpool = (
            [r for r in old_im["candidates"] if _valid_rej(r, 2, 6)]
            or [r for r in old_im["candidates"] if _valid_rej(r, 2, -12)]
            or [r for r in old_im["candidates"] if _valid_rej(r, 1, -12)]
        )
        rej = None
        if rpool:
            fc = reject_features(ch, mf[iid + "|" + ch["id"]])

            def _rscore(r):
                fr = reject_features(r, mf[iid + "|" + r["id"]])
                bs = linear(reject_models[role]["model"], _concat_diff(fr, fc))
                return bs + (-0.45 if ":s0:" in r["id"] else 0.0)

            rej = max(rpool, key=_rscore)

        aud = dict(
            role=role,
            chosen_id=ch["id"],
            rejected_id=None if rej is None else rej["id"],
            chosen_opt_meta=info[ch["id"]],
        )
        if rej is None:
            aud["reason"] = "no_eligible_hard_reject"
        else:
            pur = evaluate(ch, rej)
            aud.update(purity=pur, reason=pur["reason"])
            if pur["passed"] and ch["text"].strip() != rej["text"].strip():
                used_rej_ids.add(rej["id"])
                orig_text = orig_by_id[ch["id"]]["text"]
                img_pairs.append(
                    dict(
                        image_id=iid,
                        img_path=img_path,
                        image_sha256=img_sha,
                        prompt=PROMPT,
                        chosen=ch["text"],
                        rejected=rej["text"],
                        chosen_id=ch["id"],
                        rejected_id=rej["id"],
                        roles=[role],
                        data_policy=POLICY,
                        feature_schema="legacy_coco_v1_with_original_evidence",
                        g_fallback=False,
                        chosen_G_errors=0,
                        rejected_G_errors=len(rej["errors"]),
                        naming_gate_passed=True,
                        purity_rule=RULE,
                        purity_checks=[pur],
                        source_sha256=source_sha,
                        source_contract=source_contract,
                        repaired=(ch["text"] != orig_text),
                        original_chosen_text=orig_text,
                        opt_meta=info[ch["id"]],
                    )
                )
        c_audit.append(aud)
    return img_pairs, c_audit


def _process_chunk(args_tuple):
    chunk_items, src_data_dir, guided_map, risk_model, reject_models = args_tuple
    chunk_pairs = []
    chunk_audits = []
    chunk_sources = {}
    st = collections.Counter()

    for sh, entry in chunk_items:
        iid = str(entry["id"])
        base_path = Path(src_data_dir) / str(sh) / "results" / f"{iid}.json.gz"
        rec = read(base_path)
        if str(rec["record"]["id"]) != iid or rec["image_sha256"] != entry["image_sha256"]:
            raise ValueError(f"Record identity mismatch for {iid}")

        use_path = base_path
        old_im, new_im, use_facts, t1, t2, t3, info = extract_tiered_pools(rec)

        if not t1 and iid in guided_map:
            g_path = Path(guided_map[iid])
            grec = read(g_path)
            if str(grec["record"]["id"]) == iid and grec["image_sha256"] == entry["image_sha256"]:
                go, gn, gfs, gt1, gt2, gt3, ginf = extract_tiered_pools(grec)
                if gt1 or (not t2 and gt2):
                    use_path = g_path
                    rec = grec
                    old_im, new_im, use_facts, t1, t2, t3, info = go, gn, gfs, gt1, gt2, gt3, ginf
                    st["guided_overlay_used"] += 1

        chunk_sources[entry["id"]] = dict(
            path=str(use_path),
            sha256=sha256(use_path),
            source=rec["source_file"],
            source_sha256=rec["source_sha256"],
        )
        st["source_images"] += 1

        if len(t1) >= 1:
            pool = t1
            tier_tag = "tier1_gold"
        elif len(t2) >= 1:
            pool = t2
            tier_tag = "tier2_silver"
        elif len(t3) >= 1:
            pool = t3
            tier_tag = "tier3_bronze"
        else:
            pool = []
            tier_tag = "none"

        st[f"pool_{tier_tag}"] += 1
        img_pairs, c_audit = ([], [])
        if pool:
            img_pairs, c_audit = select_opt_pairs_for_image(
                iid=iid,
                img_path=rec["record"]["image"],
                img_sha=entry["image_sha256"],
                source_sha=rec["source_sha256"],
                source_contract=rec["source_contract"],
                old_im=old_im,
                new_im=new_im,
                facts=use_facts,
                pool=pool,
                info=info,
                risk_model=risk_model,
                reject_models=reject_models,
            )

        st["images_without_pairs"] += int(len(img_pairs) == 0)
        st["raw_proposed_pairs"] += len(img_pairs)
        chunk_pairs.extend(img_pairs)
        chunk_audits.append(
            dict(
                id=iid,
                tier=tier_tag,
                pairs=len(img_pairs),
                eligible=len(pool),
                skip=None if img_pairs else ("no_clean_unchopped_candidate" if not pool else "no_strict_pair"),
                construction=c_audit,
            )
        )

    return chunk_pairs, chunk_audits, chunk_sources, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    ap.add_argument("--src-run", type=str, default="full10000_clean_v3")
    ap.add_argument("--guided-run", type=str, default="full10000_v3p_guided")
    ap.add_argument("--out-run", type=str, default="full10000_v3_opt")
    ap.add_argument("--pair-name", type=str, default="E3_V3_OPT.json")
    ap.add_argument("--run-group", type=str, default="E3_V3_OPT")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    R = a.v3_root.resolve()
    manifest_path = R / "manifests/source_index.json"
    heldout_path = R / "evaluation/heldout_image_manifest_all.json"
    pixel_path = R / "checks/training_image_pixel_hashes.json"
    src_data_dir = R / "data" / a.src_run
    guided_dir = R / "data" / a.guided_run

    entries = read(manifest_path)
    if len(entries) != 10000:
        raise ValueError(f"Expected 10000 source entries, got {len(entries)}")

    held = read(heldout_path)
    px = read(pixel_path)
    held_sha = {x["sha256"] for x in held}
    held_rgb = {x["decoded_rgb_sha256"] for x in held}
    if any(x["sha256"] in held_sha or x["decoded_rgb_sha256"] in held_rgb for x in px):
        raise RuntimeError("Overlap detected between training_image_pixel_hashes and heldout manifest!")

    guided_map = {}
    if guided_dir.is_dir():
        for p in sorted(guided_dir.glob("*/results/*.json.gz")):
            guided_map[p.name[:-8]] = str(p)

    risk_model = read(ROOT / "models/results_v1_2.json")["risk_model"]
    reject_models = read(ROOT / "models/hc_reject_models.json")

    flat_items = []
    for sh in range(4):
        for ent in entries[sh::4]:
            flat_items.append((sh, ent))

    n_workers = max(1, min(a.workers, 32))
    chunk_size = math.ceil(len(flat_items) / n_workers)
    chunks = [
        (flat_items[i : i + chunk_size], str(src_data_dir), guided_map, risk_model, reject_models)
        for i in range(0, len(flat_items), chunk_size)
    ]

    all_pairs, all_audits, all_sources = [], [], {}
    stats = collections.Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for cp, ca, cs, st in ex.map(_process_chunk, chunks):
            all_pairs.extend(cp)
            all_audits.extend(ca)
            all_sources.update(cs)
            stats.update(st)

    if len(all_sources) != 10000:
        raise ValueError(f"Expected 10000 source artifacts, got {len(all_sources)}")

    unique = {}
    for row in all_pairs:
        key = (row["image_id"], row["prompt"], row["chosen"].strip(), row["rejected"].strip())
        if key in unique:
            unique[key]["roles"] = sorted(set(unique[key]["roles"] + row["roles"]))
            unique[key]["purity_checks"] += row["purity_checks"]
        else:
            unique[key] = dict(row)

    clean_rows = normalized_weights(list(unique.values()))
    if not clean_rows:
        raise RuntimeError("No clean pairs produced!")

    # 校验所有源契约一致性与每图精确等权归一化
    base_contract = clean_rows[0]["source_contract"]
    img_w_sums = collections.defaultdict(float)
    for r in clean_rows:
        if r["source_contract"] != base_contract:
            raise ValueError(f"Source contract mismatch on image {r['image_id']}")
        img_w_sums[r["image_id"]] += float(r["pair_weight"])
    w_spread = max(img_w_sums.values()) - min(img_w_sums.values())
    if w_spread > 1e-9:
        raise RuntimeError(f"Per-image weight spread violation: {w_spread}")

    cw_list = [len(r["chosen"].split()) for r in clean_rows]
    rw_list = [len(r["rejected"].split()) for r in clean_rows]
    cuts = [r["opt_meta"]["cut"] for r in clean_rows]
    dvs = [r["opt_meta"]["dv_div"] for r in clean_rows]
    c_s0 = sum(1 for r in clean_rows if ":s0:" in r["chosen_id"])
    r_s0 = sum(1 for r in clean_rows if ":s0:" in r["rejected_id"])
    n_1sup_rows = sum(1 for r in clean_rows if r["opt_meta"]["n_1sup"] > 0)
    cnt_1sup_rows = sum(1 for r in clean_rows if r["opt_meta"]["cnt_1sup"] > 0)

    stats["training_pairs"] = len(clean_rows)
    stats["training_images"] = len(img_w_sums)
    stats["chosen_or"] = sum(bool(OR_RE.search(r["chosen"])) for r in clean_rows)
    stats["chosen_neg"] = sum(bool(NEG_RE.search(r["chosen"])) for r in clean_rows)
    stats["cut_ge_20w"] = sum(1 for c in cuts if c >= 20)
    stats["repaired_chosen"] = sum(bool(r["repaired"]) for r in clean_rows)
    stats["chosen_words_total"] = sum(cw_list)
    stats["rejected_words_total"] = sum(rw_list)
    stats["chosen_G_errors"] = sum(r["chosen_G_errors"] for r in clean_rows)
    stats["rejected_G_errors_total"] = sum(r["rejected_G_errors"] for r in clean_rows)

    if stats["chosen_or"] != 0 or stats["chosen_neg"] != 0 or stats["cut_ge_20w"] != 0 or stats["chosen_G_errors"] != 0:
        raise RuntimeError(f"Strict zero-leak / non-chopping assertion failed: {dict(stats)}")

    code_hashes = {
        str(p.relative_to(ROOT)): sha256(p)
        for d in ["v2", "generation/rlsfv", "preference", "scripts"]
        for p in sorted((ROOT / d).glob("*.py"))
    }

    out_pairs_dir = R / "data" / a.out_run / "training_clean/pairs"
    out_pairs_dir.mkdir(parents=True, exist_ok=True)
    pair_file = out_pairs_dir / a.pair_name
    write(pair_file, clean_rows)
    write(out_pairs_dir / "construction_audit.json", all_audits)
    write(out_pairs_dir / "source_artifacts.json", all_sources)

    summary_payload = dict(
        pairs=len(clean_rows),
        images=len(img_w_sums),
        steps_2epochs_batch32=(len(clean_rows) * 2) // 32,
        per_image_weight_spread=w_spread,
        chosen_or=stats["chosen_or"],
        chosen_neg=stats["chosen_neg"],
        cut_ge_20w=stats["cut_ge_20w"],
        avg_cut_words=round(statistics.mean(cuts), 3),
        chosen_words_avg=round(statistics.mean(cw_list), 2),
        rejected_words_avg=round(statistics.mean(rw_list), 2),
        word_diff_avg=round(statistics.mean(cw_list) - statistics.mean(rw_list), 2),
        chosen_s0_pct=round(c_s0 / len(clean_rows) * 100.0, 2),
        rejected_s0_pct=round(r_s0 / len(clean_rows) * 100.0, 2),
        dv_div_avg=round(statistics.mean(dvs), 3),
        one_sup_concrete_pct=round(n_1sup_rows / len(clean_rows) * 100.0, 2),
        one_sup_count_pct=round(cnt_1sup_rows / len(clean_rows) * 100.0, 2),
        rejected_G_errors_avg=round(stats["rejected_G_errors_total"] / len(clean_rows), 3),
        pair_file=str(pair_file),
        pair_sha256=sha256(pair_file),
    )
    write(R / "data" / a.out_run / "training_clean/summary.json", summary_payload)

    prov = dict(
        policy=POLICY,
        usage="training",
        variant=a.run_group,
        source_image_count=10000,
        heldout_duplicate_source_images=0,
        feature_schema="legacy_coco_v1_with_original_evidence",
        purity_rule=RULE,
        feedback_backend="same_original_model",
        original_source_contract=base_contract,
        code=code_hashes,
        source_index_sha256=sha256(manifest_path),
        outputs={
            n: sha256(out_pairs_dir / n)
            for n in [a.pair_name, "construction_audit.json", "source_artifacts.json"]
        },
        counts=dict(stats),
        summary=summary_payload,
    )
    write(out_pairs_dir / "provenance.json", prov)

    base_train_cfg = read(R / "training_handoff/a100_training.json")
    opt_cfg = dict(
        base_train_cfg,
        pairs=str(pair_file),
        run_group=a.run_group,
        checkpoint_policy="steps_150_plus_final",
        save_steps=200,
    )
    cfg_out = R / f"training_handoff/a100_training_{a.run_group.lower()}.json"
    write(cfg_out, opt_cfg)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
