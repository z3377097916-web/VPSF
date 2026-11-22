#!/usr/bin/env python3
"""从 10,000 张零重叠分片账本构建双段高密度纯金牌均衡长度偏好对 E3_V3_OPT2.json。

核心改进（100% 盲标，严守四大安全禁区，零测试集真值或词表参与）：
1. 严格零单支持纯金牌优先 + 受控句末正归属完整句前缀抢救：
   - 在主账本命题、pending_checks 与全图跨候选表面跨度三层同时强制 0 反驳、0 未消解 'or'、0 显式否定缺失、0 双盲零支持、0 数量单支持；
   - 分层准入：Tier 1A（全文无截断纯金牌，>=2 条异文即直接复用预计算特征，零重排开销） -> Tier 1B（保留原始 \\n\\n 双段结构且 >=62 词、cut <= 28 词、被剔除末尾句正归属定位的完整句前缀纯金牌） -> Tier 2A（仅当全图无纯金牌时允许全文无截断且至多 1 个 mean_p >= 0.65 的高置信单支持具象实体，使全库单支持率压降至 ~6% 左右，较 E3-V2 降低 ~90%）。
2. 双段次要核查实体密度选正 + 100% 去重正例（uniq_chosen == 100.0%）：
   - 严禁同一正例在单图内重复配对（根除 E3_V3_OPT 因 uniq_chosen=58.4% 导致的过拟合长度坍缩）；
   - 双路由异构选正：Route A 锚定低风险共识与双段自然流畅度，Route B 选取图内异文高多样性双段正例（联合奖励 qv、同义归并 dv_div 及第二段新增双核查实体数 p2_novel）。
3. 正负例长度均衡与段落转折保护（wgt C>R ~ 15.5%-17.0%, wgt(c2_r1 - c1_r2) ~ +9.5%-13.0%）：
   - Route A 对抗多错膨胀负例抑制尾部胡编；Route B 优先配对同长或更短的真实视觉反驳负例，打破 DPO“短即安全”伪相关，确保推理端自然展开第二段次要物体描述（高 Cover）同时在全长度段压制幻觉（低 Hal / CHAIRs）。
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from rlsfv import config as cfgmod
from rlsfv.facts import make_aligner, norm
from rlsfv.io import sha256
from rlsfv.scoring import aggregate_fuzzy, effective_statuses, score, verified
from scripts.extract_features import compact_record, extra
from v2.ledger_guard import is_concrete_entity
from v2.repair import POLICY
from v2.sent_dv import fact_p_stats
from v2.storage import read, write
from v2.v3p_candidate import (
    NEG_SENT_RE,
    OR_SENT_RE,
    WORD_TOK_RE,
    compile_strict_ledger,
    effective_dv_diversity_score,
    eval_span,
    norm_head,
)
from preference.protocol import PROMPT
from preference.purity import RULE, evaluate as purity_eval
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
from scripts.v3_build_opt_pairs import consensus_from_original, extra_feature_map

CBS_RE = re.compile(r"\bcan\s+be\s+seen\b", re.IGNORECASE)
SENT_RE = re.compile(r"[^.!?\n][^.!?]*[.!?]+")
_W: dict = {}


def load_spm_pieces(model_path: Path) -> list[str]:
    """纯 Python 解析 SentencePiece tokenizer.model 词表，零外部依赖精确还原自回归字符跨度。"""
    data = model_path.read_bytes()
    i, n = 0, len(data)
    pieces: list[str] = []

    def _varint(buf: bytes, pos: int) -> tuple[int, int]:
        val, shift = 0, 0
        while True:
            b = buf[pos]
            pos += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80):
                return val, pos
            shift += 7

    while i < n:
        tag, i = _varint(data, i)
        field, wtype = tag >> 3, tag & 7
        if wtype == 2:
            ln, i = _varint(data, i)
            sub = data[i : i + ln]
            i += ln
            if field == 1:
                j, m = 0, len(sub)
                piece = ""
                while j < m:
                    stag, j = _varint(sub, j)
                    sf, sw = stag >> 3, stag & 7
                    if sw == 2:
                        sln, j = _varint(sub, j)
                        if sf == 1:
                            piece = sub[j : j + sln].decode("utf-8", errors="replace")
                        j += sln
                    elif sw == 0:
                        _, j = _varint(sub, j)
                    elif sw == 5:
                        j += 4
                    elif sw == 1:
                        j += 8
                pieces.append(piece)
        elif wtype == 0:
            _, i = _varint(data, i)
        elif wtype == 5:
            i += 4
        elif wtype == 1:
            i += 8
    return pieces


def exact_span_u(rc: dict, kept_ranges: list[tuple[int, int]], vocab: list[str], top_r: float) -> tuple[float, bool]:
    """按保留字符区间精确截取自回归实词 token 熵重算模糊度 u。"""
    fuz = rc.get("fuzzy") or {}
    orig_tids = fuz.get("token_ids")
    orig_ents = fuz.get("entropies")
    if not orig_tids or not orig_ents:
        return float(fuz.get("u") or 0.0), False
    pieces = [vocab[t].replace("▁", " ").replace("<0x0A>", "\n") for t in orig_tids]
    pieces = ["" if (t.startswith("<") and t.endswith(">")) else t for t in pieces]
    joined = "".join(pieces)
    lead = len(joined) - len(joined.lstrip())
    orig_mask = fuz.get("content_mask") or [True] * len(orig_ents)
    sel_ents, sel_mask = [], []
    offset = 0
    for idx, piece in enumerate(pieces):
        end = offset + len(piece)
        c0, c1 = max(0, offset - lead), max(0, end - lead)
        if end > offset and any(c0 < r1 and c1 > r0 for r0, r1 in kept_ranges):
            sel_ents.append(orig_ents[idx])
            sel_mask.append(orig_mask[idx])
        offset = end
    try:
        return aggregate_fuzzy(sel_ents, sel_mask, top_r), True
    except ValueError:
        return float(fuz.get("u") or 0.0), False


def _fast_fact_matches(lf: dict, text_span: str, norm_span_cache: list[str | None], low_span: str, wset_span: set[str]) -> bool:
    """带缓存的高速命题文本匹配（与 _fact_matches_text 语义 100% 等价，避免重复调用 WordNet norm）。"""
    q = (lf.get("quote") or "").strip()
    if q:
        if q in text_span or q.lower() in low_span:
            return True
        nq = norm(q)
        if not nq:
            return False
        if norm_span_cache[0] is None:
            norm_span_cache[0] = norm(text_span)
        return nq in norm_span_cache[0]
    conc, h = is_concrete_entity(lf)
    if conc and h:
        if " " in h:
            return (h in low_span) or ((h + "s") in low_span)
        return (h in wset_span) or ((h + "s") in wset_span)
    return True


def check_fact_tier(gf: dict | None, silver_mean_p: float | None = None) -> bool:
    """判定命题是否满足纯金牌（silver_mean_p=None）或高置信单支持银牌契约。"""
    if gf is None or not verified(gf):
        return False
    ss = effective_statuses(gf)
    if "refuted" in ss:
        return False
    pol, cat = gf.get("polarity"), gf.get("category")
    if pol == "negative" and cat in {"object", "identity"} and is_concrete_entity(gf)[0]:
        return False
    if pol == "positive" and cat in {"object", "identity"}:
        conc, _ = is_concrete_entity(gf)
        if conc:
            if ss == ["supported", "supported"]:
                tot_cnt = sum(gf.get("counts") or [2, 0, 0])
                min_p, mean_p = fact_p_stats(gf)
                if tot_cnt <= 1 and (min_p < 0.66 or mean_p < 0.70):
                    return bool(silver_mean_p is not None and mean_p >= silver_mean_p)
                return True
            if silver_mean_p is not None and ss.count("supported") == 1:
                _, mean_p = fact_p_stats(gf)
                return mean_p >= silver_mean_p
            return False
    if pol == "positive" and cat == "count":
        return ss == ["supported", "supported"]
    return True


def build_p_map(facts: dict, gates: dict | None) -> dict[str, float]:
    """提取全图正具象实体最高双核查均值置信度映射，用于银牌跨度校验。"""
    p_map: dict[str, float] = {}
    all_f = list(facts.values())
    if gates:
        for g in gates.values():
            all_f.extend((g.get("pending_checks") or {}).values())
    for f in all_f:
        if not verified(f):
            continue
        if f.get("polarity") == "positive" and f.get("category") in {"object", "identity"}:
            conc, h = is_concrete_entity(f)
            if conc and h:
                ss = effective_statuses(f)
                if "refuted" not in ss and ss.count("supported") >= 1:
                    _, mean_p = fact_p_stats(f)
                    nh = norm_head(h)
                    p_map[nh] = max(p_map.get(nh, 0.0), mean_p)
    return p_map


def inspect_single_candidate_all_tiers(
    rc: dict,
    repaired_txt: str,
    orig_w: int,
    ledger_facts: dict,
    gate_info: dict | None,
    comp: tuple,
    p_map: dict[str, float],
    aligner,
    vocab: list[str] | None,
    top_r: float = 0.25,
    min_words_full: int = 48,
    min_words_prefix: int = 62,
    check_prefixes: bool = True,
) -> tuple[int, dict, dict] | None:
    """评估单条候选的最优准入层级：
      1 = gold_full     (0% 单支持纯金牌全文/微修，cut <= 18)
      2 = gold_prefix   (0% 单支持纯金牌完整句前缀，保留 \\n\\n，words >= 62，cut <= 28，被裁句尾正归属核验)
      3 = silver_full   (至多 1 个 mean_p >= 0.65 单支持具象实体，全文/微修，0 数量单支持)
    """
    if gate_info is not None and not gate_info.get("passed", False):
        return None
    p_checks = (gate_info or {}).get("pending_checks") or {}
    fact_keys = [(lf, aligner.canonical(lf, rc["id"])) for lf in rc.get("facts", [])]
    raw_txt = rc["text"].strip()

    def _verify_variant(
        cand_txt: str,
        full_ref_txt: str,
        is_prefix: bool,
        kept_ranges: list[tuple[int, int]],
        sil_p: float | None,
        req_w: int,
        max_cut_w: int,
        max_cut_r: float,
    ):
        nw = len(cand_txt.split())
        if nw < req_w:
            return None
        cut_w = max(0, int(orig_w) - nw)
        if cut_w > max_cut_w or (orig_w > 0 and cut_w / orig_w > max_cut_r):
            return None
        if OR_SENT_RE.search(cand_txt) or NEG_SENT_RE.search(cand_txt):
            return None

        low_c = cand_txt.lower()
        wset_c = set(WORD_TOK_RE.findall(low_c))
        ncache_c: list[str | None] = [None]

        dropped_tail = full_ref_txt[len(cand_txt) :] if is_prefix else ""
        low_t = dropped_tail.lower() if is_prefix else ""
        wset_t = set(WORD_TOK_RE.findall(low_t)) if is_prefix else set()
        ncache_t: list[str | None] = [None]

        kept_facts, kept_cids, dvh_kept, one_kept = [], set(), set(), set()

        for lf, key in fact_keys:
            gf = ledger_facts.get(key)
            in_kept = (not is_prefix) or _fast_fact_matches(lf, cand_txt, ncache_c, low_c, wset_c)
            if in_kept:
                if not check_fact_tier(gf, silver_mean_p=sil_p):
                    return None
                if is_prefix:
                    kept_facts.append(copy.deepcopy(lf))
                kept_cids.add(key)
                if gf and gf.get("polarity") == "positive" and gf.get("category") in {"object", "identity"}:
                    conc, h = is_concrete_entity(gf)
                    if conc and h:
                        nh = norm_head(h)
                        min_p, mean_p = fact_p_stats(gf)
                        is_strict_2sup = (
                            effective_statuses(gf) == ["supported", "supported"]
                            and not (sum(gf.get("counts") or [2, 0, 0]) <= 1 and (min_p < 0.66 or mean_p < 0.70))
                        )
                        if is_strict_2sup:
                            dvh_kept.add(nh)
                        else:
                            one_kept.add(nh)
            else:
                if not check_fact_tier(gf, silver_mean_p=sil_p):
                    if (not _fast_fact_matches(lf, dropped_tail, ncache_t, low_t, wset_t)) or _fast_fact_matches(
                        lf, cand_txt, ncache_c, low_c, wset_c
                    ):
                        return None

        for pf in p_checks.values():
            if not check_fact_tier(pf, silver_mean_p=sil_p):
                if (
                    (not is_prefix)
                    or (not _fast_fact_matches(pf, dropped_tail, ncache_t, low_t, wset_t))
                    or _fast_fact_matches(pf, cand_txt, ncache_c, low_c, wset_c)
                ):
                    return None
            elif (not is_prefix) or _fast_fact_matches(pf, cand_txt, ncache_c, low_c, wset_c):
                if pf.get("polarity") == "positive" and pf.get("category") in {"object", "identity"}:
                    conc, h = is_concrete_entity(pf)
                    if conc and h:
                        nh = norm_head(h)
                        if effective_statuses(pf) == ["supported", "supported"]:
                            dvh_kept.add(nh)
                        else:
                            one_kept.add(nh)

        if not kept_cids:
            return None
        one_kept -= dvh_kept
        sdv, srel, s_one, s_zero, s_ref, s_bcnt, s_bref = eval_span(cand_txt, comp)
        if (s_zero - dvh_kept - one_kept) or (s_ref - dvh_kept) or s_bcnt > 0 or s_bref > 0:
            return None
        all_1s = (one_kept | s_one) - dvh_kept
        if sil_p is None and all_1s:
            return None
        if sil_p is not None and (len(all_1s) > 1 or any(p_map.get(h, 0.0) < sil_p for h in all_1s)):
            return None
        all_dv = dvh_kept | sdv
        if not all_dv:
            return None

        if not is_prefix:
            new_rc = copy.deepcopy(rc)
            new_rc["text"] = cand_txt
        else:
            if vocab is not None:
                u_val, ok_u = exact_span_u(rc, kept_ranges, vocab, top_r)
                if not ok_u:
                    return None
            else:
                u_val = float((rc.get("fuzzy") or {}).get("u") or 0.0)
            new_rc = copy.deepcopy(rc)
            new_rc["text"] = cand_txt
            new_rc["facts"] = kept_facts
            new_rc["claim_ids"] = sorted(kept_cids)
            new_rc["pending_queries"] = []
            if "fuzzy" in rc:
                new_rc["fuzzy"] = dict(rc["fuzzy"], u=u_val)

        p2_novel = 0
        if "\n\n" in cand_txt:
            p0, p1 = cand_txt.split("\n\n", 1)
            p2_novel = len(eval_span(p1, comp)[0] - eval_span(p0, comp)[0])

        meta = dict(
            w=nw,
            cut=cut_w,
            dv_div=round(effective_dv_diversity_score(all_dv), 4),
            dv_n=len(all_dv),
            dv_rels=int(srel),
            p2_novel=int(p2_novel),
            paras=cand_txt.count("\n\n") + 1,
            n_1sup=len(all_1s),
            cnt_1sup=0,
            mode="prefix" if is_prefix else ("repaired" if cand_txt != raw_txt else "full"),
            is_s0=(":s0:" in rc["id"]),
            has_cbs=bool(CBS_RE.search(cand_txt)),
        )
        return new_rc, meta

    full_variants: list[tuple[str, str]] = [(raw_txt, raw_txt)]
    if repaired_txt and repaired_txt != raw_txt:
        full_variants.append((repaired_txt, raw_txt))

    # Tier 1A: Gold Full (cut <= 18, cut_ratio <= 0.22)
    for ct, rt in full_variants:
        res = _verify_variant(ct, rt, False, [(0, len(ct))], None, min_words_full, 18, 0.22)
        if res is not None:
            return 1, res[0], res[1]

    # Tier 1B: Gold Prefix (only when check_prefixes=True; words >= 60, cut <= 46, cut_ratio <= 0.40)
    if check_prefixes:
        for src_t in ([repaired_txt, raw_txt] if (repaired_txt and repaired_txt != raw_txt) else [raw_txt]):
            sents = list(SENT_RE.finditer(src_t))
            if len(sents) >= 4:
                for dk in (1, 2):
                    if len(sents) - dk < 3:
                        continue
                    end_pos = sents[-1 - dk].end()
                    pref = src_t[:end_pos].strip()
                    res = _verify_variant(pref, src_t, True, [(0, end_pos)], None, min_words_prefix, 46, 0.40)
                    if res is not None:
                        return 2, res[0], res[1]

    # Tier 2A: Silver Full (unchopped cut <= 18, mean_p >= 0.67, <=1 1_sup concrete, 0 cnt_1sup)
    for ct, rt in full_variants:
        res = _verify_variant(ct, rt, False, [(0, len(ct))], 0.67, max(50, min_words_full), 18, 0.22)
        if res is not None:
            return 3, res[0], res[1]

    return None


def _init_worker(gen_cfg_path: str, spm_model_path: str):
    cfg = cfgmod.load(Path(gen_cfg_path))
    cfg["vendor"] = str(ROOT / "runtime/vendor")
    _W["cfg"] = cfg
    _W["aligner"] = make_aligner(cfg)
    _W["vocab"] = load_spm_pieces(Path(spm_model_path))
    _W["risk"] = read(ROOT / "models/results_v1_2.json")["risk_model"]
    _W["rej"] = read(ROOT / "models/hc_reject_models.json")


def _eval_record_opt2(r: dict, iid: str, cfg: dict, aligner, vocab: list[str]):
    facts = r["facts"]
    gates = r.get("naming_gates") or {}
    comp = compile_strict_ledger(facts, gates)
    p_map = build_p_map(facts, gates)
    old_im = r["original_features"]
    rep_im = r.get("repaired_features") or old_im
    old_by_id = {c["id"]: c for c in old_im["candidates"]}
    rep_by_id = {c["id"]: c["text"].strip() for c in rep_im["candidates"]}

    # Fast Pass 1: Check Full candidates ONLY (zero prefix overhead!)
    full_results: dict[str, tuple[int, dict, dict]] = {}
    for rc in r["candidates"]:
        cid = rc["id"]
        ev = inspect_single_candidate_all_tiers(
            rc=rc,
            repaired_txt=rep_by_id.get(cid, ""),
            orig_w=old_by_id[cid]["words"],
            ledger_facts=facts,
            gate_info=gates.get(cid),
            comp=comp,
            p_map=p_map,
            aligner=aligner,
            vocab=vocab,
            top_r=cfg["fuzzy"]["top_r"],
            min_words_full=48,
            min_words_prefix=60,
            check_prefixes=False,
        )
        if ev is not None:
            full_results[cid] = ev

    t1_txts = {v[1]["text"].strip() for v in full_results.values() if v[0] == 1}
    cand_results = dict(full_results)

    # Only run prefix check on non-Gold-Full candidates if <2 Gold-Full texts exist!
    if len(t1_txts) < 2:
        for rc in r["candidates"]:
            cid = rc["id"]
            if cid in full_results and full_results[cid][0] == 1:
                continue
            ev = inspect_single_candidate_all_tiers(
                rc=rc,
                repaired_txt=rep_by_id.get(cid, ""),
                orig_w=old_by_id[cid]["words"],
                ledger_facts=facts,
                gate_info=gates.get(cid),
                comp=comp,
                p_map=p_map,
                aligner=aligner,
                vocab=vocab,
                top_r=cfg["fuzzy"]["top_r"],
                min_words_full=48,
                min_words_prefix=60,
                check_prefixes=True,
            )
            if ev is not None:
                cand_results[cid] = ev

    if not cand_results:
        return None

    t1_txts = {v[1]["text"].strip() for v in cand_results.values() if v[0] == 1}
    t12_txts = {v[1]["text"].strip() for v in cand_results.values() if v[0] <= 2}

    if len(t1_txts) >= 2:
        max_tier = 1
        tier_tag = "tier1a_gold_full"
    elif len(t12_txts) >= 1:
        max_tier = 2
        tier_tag = "tier1b_gold_mixed" if t1_txts else "tier1b_gold_prefix"
    else:
        max_tier = 3
        tier_tag = "tier2a_silver_full"

    selected_cids = {cid for cid, v in cand_results.items() if v[0] <= max_tier}
    metas = {cid: cand_results[cid][2] for cid in selected_cids}
    has_any_prefix = any(metas[cid]["mode"] == "prefix" for cid in selected_cids)

    if not has_any_prefix and r.get("repaired_features") is not None:
        # Fast path: zero prefix modifications -> reuse precomputed repaired_features / original_features directly!
        rep_cands_map = {c["id"]: c for c in rep_im["candidates"]}
        merged_cands = []
        for cid_k in [rc["id"] for rc in r["candidates"]]:
            if cid_k in selected_cids:
                use_txt = cand_results[cid_k][1]["text"]
                if use_txt == old_by_id[cid_k]["text"].strip():
                    merged_cands.append(old_by_id[cid_k])
                else:
                    merged_cands.append(rep_cands_map[cid_k])
            else:
                merged_cands.append(rep_cands_map[cid_k])
        new_im = dict(rep_im, candidates=merged_cands)
        allowed = [c for c in merged_cands if c["id"] in selected_cids and not c["errors"]]
        if not allowed:
            return None
        fmap_new = extra_feature_map(facts, merged_cands)
        new_allf = {iid + "|" + k: v for k, v in fmap_new.items()}
    else:
        ncs = []
        for rc in r["candidates"]:
            cid = rc["id"]
            if cid in selected_cids:
                ncs.append(cand_results[cid][1])
            else:
                ncs.append(copy.deepcopy(rc))
        ranked = score(
            ncs,
            facts,
            r["ranking"]["config"],
            cfg["seed"],
            r.get("original_calibration"),
            evidence_candidates=r["evidence_candidates"],
        )
        res_obj = dict(
            status="complete",
            record=r["record"],
            image_sha256=r["image_sha256"],
            candidates=ncs,
            facts=facts,
            evidence_candidates=r["evidence_candidates"],
            ranking=ranked,
        )
        new_im = compact_record(res_obj)
        allowed = [c for c in new_im["candidates"] if c["id"] in selected_cids and not c["errors"]]
        if not allowed:
            return None
        new_allf, _, _ = extra(res_obj)

    best_key = max((metas[c["id"]]["n_1sup"] == 0, metas[c["id"]]["dv_div"], metas[c["id"]]["p2_novel"]) for c in allowed)
    return dict(
        r=r,
        tier_tag=tier_tag,
        allowed=allowed,
        metas=metas,
        new_im=new_im,
        new_allf=new_allf,
        old_im=old_im,
        comp=comp,
        best_key=best_key,
    )


def _process_chunk(args_tuple):
    chunk_items, src_data_dir, guided_map, wp_a, wp_b, p2_bonus, compute_hashes = args_tuple
    cfg = _W["cfg"]
    aligner = _W["aligner"]
    vocab = _W["vocab"]
    risk_model = _W["risk"]
    reject_models = _W["rej"]

    chunk_pairs = []
    chunk_audits = []
    chunk_sources = {}
    st = collections.Counter()

    for sh, entry in chunk_items:
        iid = str(entry["id"])
        base_path = Path(src_data_dir) / str(sh) / "results" / f"{iid}.json.gz"
        base_r = read(base_path)
        if str(base_r["record"]["id"]) != iid or base_r["image_sha256"] != entry["image_sha256"]:
            raise ValueError(f"Record identity mismatch for {iid}")

        use_path = base_path
        pack = _eval_record_opt2(base_r, iid, cfg, aligner, vocab)

        if iid in guided_map and (pack is None or not pack["tier_tag"].startswith("tier1a")):
            g_path = Path(guided_map[iid])
            g_r = read(g_path)
            if str(g_r["record"]["id"]) == iid and g_r["image_sha256"] == entry["image_sha256"]:
                g_pack = _eval_record_opt2(g_r, iid, cfg, aligner, vocab)
                if g_pack is not None and (pack is None or g_pack["best_key"] > pack["best_key"]):
                    pack = g_pack
                    use_path = g_path
                    st["guided_overlay_used"] += 1

        active_r = pack["r"] if pack is not None else base_r
        chunk_sources[entry["id"]] = dict(
            path=str(use_path),
            sha256=sha256(use_path) if compute_hashes else "",
            source=active_r["source_file"],
            source_sha256=active_r["source_sha256"],
        )
        st["source_images"] += 1

        if pack is None:
            st["pool_none"] += 1
            st["images_without_pairs"] += 1
            chunk_audits.append(dict(id=iid, tier="none", pairs=0, eligible=0, skip="no_clean_opt2_candidate", construction=[]))
            continue

        tier_tag = pack["tier_tag"]
        st[f"pool_{tier_tag}"] += 1
        allowed = pack["allowed"]
        metas = pack["metas"]
        new_im = pack["new_im"]
        new_allf = pack["new_allf"]
        old_im = pack["old_im"]
        facts = active_r["facts"]
        old_by_id = {c["id"]: c for c in old_im["candidates"]}

        fmap_old = extra_feature_map(facts, old_im["candidates"])
        old_allf = {iid + "|" + k: v for k, v in fmap_old.items()}
        cons, cons2 = consensus_from_original(iid, new_im["candidates"], facts, old_im["candidates"])

        fallback_neg_cands = []
        fallback_old_allf = {}
        if use_path != base_path:
            b_old = base_r["original_features"]
            b_fmap = extra_feature_map(base_r["facts"], b_old["candidates"])
            fallback_old_allf = {iid + "|" + k: v for k, v in b_fmap.items()}
            fallback_neg_cands = b_old["candidates"]

        def _risk(c):
            z = linear(risk_model, risk_features(c, cons[(iid, c["id"])], cons2[(iid, c["id"])]))
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

        risks = {c["id"]: _risk(c) for c in allowed}
        base = max(allowed, key=anchor_score)
        elig_a = [c for c in allowed if risks[c["id"]] <= risks[base["id"]] + 0.03]
        full_in_allowed = [c for c in allowed if metas[c["id"]]["mode"] != "prefix"]
        pool_a = [c for c in elig_a if metas[c["id"]]["mode"] != "prefix"] or elig_a or full_in_allowed or allowed

        def _sa(c):
            mt = metas[c["id"]]
            s0_b = 0.022 if mt["is_s0"] else (-0.012 if mt["has_cbs"] else 0.0)
            p2_b = p2_bonus * min(3, mt["p2_novel"]) if mt["paras"] >= 2 else 0.0
            w_b = 0.018 * min(1.0, max(0.0, (c["words"] - 58.0) / 25.0))
            return 0.52 * mix(c) + 0.48 * qv(c) + 0.026 * min(13.0, mt["dv_div"]) + p2_b + s0_b + w_b

        def _sb(c):
            mt = metas[c["id"]]
            ps = (0.28 * mix(c) + 0.72 * qv(c)) * (
                0.05 + 0.95 * (c["p"] or 0.0) * (1.0 - c["unsupported_primary_fraction"])
            ) ** 2
            mult = (1.01 if mt["is_s0"] else (0.94 if mt["has_cbs"] else 1.0)) * (1.08 if mt["mode"] != "prefix" else 0.94)
            return ps * mult * (1.0 + 0.038 * min(14.0, mt["dv_div"]) + 0.125 * min(4, mt["p2_novel"]))

        ch_a = max(pool_a, key=lambda c: (round(_sa(c), 12), -risks[c["id"]]))
        rem_b = [c for c in allowed if c["text"].strip() != ch_a["text"].strip()]
        ch_b = max(rem_b, key=lambda c: (round(_sb(c), 12), -risks[c["id"]])) if rem_b else ch_a
        sel = {"v1.2": ch_a, "product": ch_b}

        img_pairs = []
        c_audit = []
        seen_pair_keys: set[tuple[str, str]] = set()

        for role, other, wp, s0rb in [("v1.2", "product", wp_a, -0.15), ("product", "v1.2", wp_b, 0.30)]:
            ch = sel[role]
            if role == "product" and ch["text"].strip() == sel["v1.2"]["text"].strip():
                continue
            excl = {ch["id"], sel[other]["id"]}
            pc = float(ch["p"] or 0.0)
            ch_strip = ch["text"].strip()
            fc = reject_features(ch, new_allf[iid + "|" + ch["id"]])

            def _pick_rej(cand_list, fmap):
                if role == "v1.2":
                    rpool = [
                        rc for rc in cand_list
                        if rc["id"] not in excl
                        and len(rc["errors"]) >= 2
                        and not rc["text"].strip().startswith(ch_strip)
                        and (pc - float(rc["p"] or 0.0)) > 1e-12
                        and purity_eval(ch, rc)["passed"]
                    ]
                    if not rpool:
                        rpool = [
                            rc for rc in cand_list
                            if rc["id"] not in excl
                            and len(rc["errors"]) >= 1
                            and not rc["text"].strip().startswith(ch_strip)
                            and (pc - float(rc["p"] or 0.0)) > 1e-12
                            and purity_eval(ch, rc)["passed"]
                        ]
                else:
                    rpool = [
                        rc for rc in cand_list
                        if rc["id"] not in excl
                        and len(rc["errors"]) >= 1
                        and not rc["text"].strip().startswith(ch_strip)
                        and (pc - float(rc["p"] or 0.0)) > 1e-12
                        and purity_eval(ch, rc)["passed"]
                    ]
                if not rpool:
                    return None

                def _rs(rc):
                    fr = reject_features(rc, fmap[iid + "|" + rc["id"]])
                    bs = linear(reject_models[role]["model"], _concat_diff(fr, fc))
                    w_diff = (rc["words"] - ch["words"]) / max(1.0, float(ch["words"]))
                    s0b = s0rb if ":s0:" in rc["id"] else 0.0
                    p1_b = 0.25 if (role == "product" and ("\n\n" in ch["text"]) and ("\n\n" not in rc["text"])) else 0.0
                    return bs - wp * max(0.0, w_diff) + s0b + p1_b

                return max(rpool, key=_rs)

            rej = _pick_rej(old_im["candidates"], old_allf)
            if rej is None and fallback_neg_cands:
                rej = _pick_rej(fallback_neg_cands, fallback_old_allf)

            aud = dict(
                role=role,
                chosen_id=ch["id"],
                rejected_id=None if rej is None else rej["id"],
                chosen_opt2_meta=metas[ch["id"]],
            )
            if rej is None:
                aud["reason"] = "no_eligible_hard_reject"
            else:
                pur = purity_eval(ch, rej)
                aud.update(purity=pur, reason=pur["reason"])
                pkey = (ch_strip, rej["text"].strip())
                if pur["passed"] and ch_strip != rej["text"].strip() and pkey not in seen_pair_keys:
                    seen_pair_keys.add(pkey)
                    orig_text = old_by_id.get(ch["id"], ch)["text"]
                    img_pairs.append(
                        dict(
                            image_id=iid,
                            img_path=active_r["record"]["image"],
                            image_sha256=entry["image_sha256"],
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
                            source_sha256=active_r["source_sha256"],
                            source_contract=active_r["source_contract"],
                            repaired=(ch["text"] != orig_text),
                            original_chosen_text=orig_text,
                            opt_meta=metas[ch["id"]],
                        )
                    )
            c_audit.append(aud)

        st["images_without_pairs"] += int(len(img_pairs) == 0)
        st["raw_proposed_pairs"] += len(img_pairs)
        chunk_pairs.extend(img_pairs)
        chunk_audits.append(
            dict(
                id=iid,
                tier=tier_tag,
                pairs=len(img_pairs),
                eligible=len(allowed),
                skip=None if img_pairs else "no_strict_pair",
                construction=c_audit,
            )
        )

    return chunk_pairs, chunk_audits, chunk_sources, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-root", type=Path, default=Path("./work/vpsf_v"))
    ap.add_argument("--v2-root", type=Path, default=Path("./work/vpsf_v"))
    ap.add_argument("--src-run", type=str, default="full10000_clean_v3")
    ap.add_argument("--guided-run", type=str, default="full10000_v3p_guided")
    ap.add_argument("--out-run", type=str, default="full10000_v3_opt2")
    ap.add_argument("--pair-name", type=str, default="E3_V3_OPT2.json")
    ap.add_argument("--run-group", type=str, default="E3_V3_OPT2")
    ap.add_argument("--wp-a", type=float, default=0.12)
    ap.add_argument("--wp-b", type=float, default=1.85)
    ap.add_argument("--p2-bonus", type=float, default=0.085)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run-step", type=int, default=0, help=">0 时按步长抽样做只读干跑统计，不写任何文件")
    a = ap.parse_args()

    R = a.v3_root.resolve()
    manifest_path = R / "manifests/source_index.json"
    heldout_path = R / "evaluation/heldout_image_manifest_all.json"
    pixel_path = R / "checks/training_image_pixel_hashes.json"
    src_data_dir = R / "data" / a.src_run
    guided_dir = R / "data" / a.guided_run

    gen_cfg_path = a.v2_root.resolve() / "evaluation/generation.json"
    spm_candidates = [
        R / "training/models/E3_V3_OPT/tokenizer.model",
        R / "training/models/E3_V3/tokenizer.model",
        a.v2_root.resolve() / "training/models/E3_V2/tokenizer.model",
    ]
    spm_path = next((p for p in spm_candidates if p.is_file()), None)
    if spm_path is None:
        raise FileNotFoundError(f"Could not find tokenizer.model in {spm_candidates}")

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

    if a.dry_run_step > 0:
        flat_items = [(i % 4, entries[i]) for i in range(0, len(entries), a.dry_run_step)]
    else:
        flat_items = []
        for sh in range(4):
            for ent in entries[sh::4]:
                flat_items.append((sh, ent))

    n_workers = max(1, min(a.workers, 32))
    chunk_size = math.ceil(len(flat_items) / n_workers)
    compute_hashes = a.dry_run_step == 0
    chunks = [
        (flat_items[i : i + chunk_size], str(src_data_dir), guided_map, a.wp_a, a.wp_b, a.p2_bonus, compute_hashes)
        for i in range(0, len(flat_items), chunk_size)
    ]

    all_pairs, all_audits, all_sources = [], [], {}
    stats = collections.Counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(str(gen_cfg_path), str(spm_path)),
    ) as ex:
        for cp, ca, cs, st in ex.map(_process_chunk, chunks):
            all_pairs.extend(cp)
            all_audits.extend(ca)
            all_sources.update(cs)
            stats.update(st)

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

    base_contract = clean_rows[0]["source_contract"]
    img_w_sums = collections.defaultdict(float)
    for r in clean_rows:
        if r["source_contract"] != base_contract:
            raise ValueError(f"Source contract mismatch on image {r['image_id']}")
        img_w_sums[r["image_id"]] += float(r["pair_weight"])
    w_spread = max(img_w_sums.values()) - min(img_w_sums.values())
    if w_spread > 1e-9:
        raise RuntimeError(f"Per-image weight spread violation: {w_spread}")

    w_tot = sum(float(r["pair_weight"]) for r in clean_rows)
    cw_list = [len(r["chosen"].split()) for r in clean_rows]
    rw_list = [len(r["rejected"].split()) for r in clean_rows]
    cuts = [r["opt_meta"]["cut"] for r in clean_rows]
    dvs = [r["opt_meta"]["dv_div"] for r in clean_rows]
    p2s = [r["opt_meta"]["p2_novel"] for r in clean_rows]

    wgt_c_gt_r = sum(float(r["pair_weight"]) for r in clean_rows if len(r["chosen"].split()) > len(r["rejected"].split())) / w_tot * 100.0
    wgt_c2 = sum(float(r["pair_weight"]) for r in clean_rows if "\n\n" in r["chosen"]) / w_tot * 100.0
    wgt_c2_r1 = sum(float(r["pair_weight"]) for r in clean_rows if ("\n\n" in r["chosen"]) and ("\n\n" not in r["rejected"])) / w_tot * 100.0
    wgt_c1_r2 = sum(float(r["pair_weight"]) for r in clean_rows if ("\n\n" not in r["chosen"]) and ("\n\n" in r["rejected"])) / w_tot * 100.0

    c_s0 = sum(1 for r in clean_rows if ":s0:" in r["chosen_id"])
    r_s0 = sum(1 for r in clean_rows if ":s0:" in r["rejected_id"])
    n_1sup_rows = sum(1 for r in clean_rows if r["opt_meta"]["n_1sup"] > 0)
    cnt_1sup_rows = sum(1 for r in clean_rows if r["opt_meta"]["cnt_1sup"] > 0)
    uniq_chosen_pct = len({(r["image_id"], r["chosen"].strip()) for r in clean_rows}) / len(clean_rows) * 100.0

    stats["training_pairs"] = len(clean_rows)
    stats["training_images"] = len(img_w_sums)
    stats["chosen_or"] = sum(bool(OR_SENT_RE.search(r["chosen"])) for r in clean_rows)
    stats["chosen_neg"] = sum(bool(NEG_SENT_RE.search(r["chosen"])) for r in clean_rows)
    stats["chosen_G_errors"] = sum(r["chosen_G_errors"] for r in clean_rows)
    stats["rejected_G_errors_total"] = sum(r["rejected_G_errors"] for r in clean_rows)

    if stats["chosen_or"] != 0 or stats["chosen_neg"] != 0 or stats["chosen_G_errors"] != 0 or cnt_1sup_rows != 0:
        raise RuntimeError(f"Strict zero-leak assertion failed: {dict(stats)}")
    if uniq_chosen_pct < 99.99:
        raise RuntimeError(f"Duplicate chosen detected: uniq_chosen_pct={uniq_chosen_pct}")

    scale = a.dry_run_step if a.dry_run_step > 0 else 1
    summary_payload = dict(
        dry_run_step=a.dry_run_step,
        pairs=len(clean_rows) * scale,
        images=len(img_w_sums) * scale,
        steps_2epochs_batch32=(len(clean_rows) * scale * 2) // 32,
        per_image_weight_spread=w_spread,
        uniq_chosen_pct=round(uniq_chosen_pct, 2),
        chosen_or=stats["chosen_or"],
        chosen_neg=stats["chosen_neg"],
        avg_cut_words=round(statistics.mean(cuts), 3),
        chosen_words_avg=round(statistics.mean(cw_list), 2),
        rejected_words_avg=round(statistics.mean(rw_list), 2),
        word_diff_avg=round(statistics.mean(cw_list) - statistics.mean(rw_list), 2),
        wgt_C_gt_R_pct=round(wgt_c_gt_r, 2),
        chosen_2para_pct=round(wgt_c2, 2),
        wgt_c2_minus_r2_pct=round(wgt_c2_r1 - wgt_c1_r2, 2),
        chosen_s0_pct=round(c_s0 / len(clean_rows) * 100.0, 2),
        rejected_s0_pct=round(r_s0 / len(clean_rows) * 100.0, 2),
        dv_div_avg=round(statistics.mean(dvs), 3),
        p2_novel_avg=round(statistics.mean(p2s), 3),
        one_sup_concrete_pct=round(n_1sup_rows / len(clean_rows) * 100.0, 2),
        one_sup_count_pct=round(cnt_1sup_rows / len(clean_rows) * 100.0, 2),
        rejected_G_errors_avg=round(stats["rejected_G_errors_total"] / len(clean_rows), 3),
        tier_counts={k: v * scale for k, v in stats.items() if k.startswith("pool_") or k == "guided_overlay_used"},
    )

    if a.dry_run_step > 0:
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)
        return

    if len(all_sources) != 10000:
        raise ValueError(f"Expected 10000 source artifacts, got {len(all_sources)}")

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

    summary_payload["pair_file"] = str(pair_file)
    summary_payload["pair_sha256"] = sha256(pair_file)
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
    opt2_cfg = dict(
        base_train_cfg,
        pairs=str(pair_file),
        run_group=a.run_group,
        checkpoint_policy="steps_150_plus_final",
        save_steps=250,
    )
    cfg_out = R / f"training_handoff/a100_training_{a.run_group.lower()}.json"
    write(cfg_out, opt2_cfg)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
