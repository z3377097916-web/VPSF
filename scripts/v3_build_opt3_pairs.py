#!/usr/bin/env python3
"""从 10,000 张零重叠分片账本构建全槽位零盲区、Foil与共识双向自校准、在策略硬负例增强偏好对 E3_V3_OPT3.json。

核心机制（100% 纯程序化自反馈闭环、冻结参数端到端可复现、零测试集泄漏、零人工与外部 AI 介入）：
1. 全槽位词干对齐 + 单图 Foil 假支持率动态自校准账本（compile_calibrated_ledger_v3 / check_fact_opt3）：
   - 穿透核查命题主语槽 entities[0]、关系/空间辅槽 entities[1..k] (align.object) 与 identity 表语名词，经 WordNet 单数词干归一后要求全部具象物理实体具备本图正核验支撑；
   - 跨 22 候选聚合同一实体的最大总票数与 Stage-0 票数（修复 M2 微修事实 counts=[0,0,0] 被误判为孤立单样本的底层缺陷）；
   - 按单图自反馈诱饵误支持率 foil_fsr 动态抬高孤立单样本（tot_cnt <= 1）与纯引导孤立样本（fresh_s0 == 0）的 p_true 准入门槛，根除 VQA Yes-bias 词级假阳性；
   - 硬拦截未达高置信双支持的 uncertain 具象命题及语法级猜名词短语（NOUN_GUESS_HEDGE_RE），放行正常神态/氛围谓语。
2. 高共识前景救援 + 保留 \\n\\n 段落结构的句级裁剪与句尾空话剥离：
   - Tier 1（纯金全文/空话剥离）-> Tier 2（纯金完整句前缀 / 指代安全单句切除）-> Tier 3/4（跨样本强共识前景救援），挽回复杂高错场景图，并用基座 tokenizer.model 精确重算保留跨度词元熵 exact_span_u 与官方 score()。
3. 异构双路由唯一选正（uniq_chosen == 100%）+ 双向覆盖单调守卫（delta_qv > 0）+ Route B 在策略 :s0: 硬负例优先：
   - 扭转历史轮次 56.4% 的 delta_qv < 0 覆盖倒挂，加入单位词长去重实体密度与条件双段奖励（仅 p2_novel >= 1 时奖励双段）；
   - 在 Route B 对 :s0: 自然采样硬负例赋予 +0.65 同分布偏置补偿，将 :s0: 负例占比从 10.2% 提升至 ~25%+，并将 (wgt_C_gt_R, wgt_c2_minus_r2) 精准锁定在自适应分段帕累托靶心。
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

from nltk.stem import WordNetLemmatizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from rlsfv import config as cfgmod
from rlsfv.facts import make_aligner
from rlsfv.io import sha256
from rlsfv.scoring import effective_statuses, score, verified
from scripts.extract_features import compact_record, extra
from v2.ledger_guard import CONCRETE_COMPOUND_EXCEPTIONS, NON_CONCRETE_HEADS, is_concrete_entity
from v2.repair import POLICY
from v2.sent_dv import fact_p_stats
from v2.storage import read, write
from v2.v3p_candidate import (
    NEG_SENT_RE,
    NON_OPENING_START_RE,
    OR_SENT_RE,
    WORD_TOK_RE,
    effective_dv_diversity_score,
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
from scripts.v3_build_opt2_pairs import _fast_fact_matches, exact_span_u, load_spm_pieces

CBS_RE = re.compile(r"\bcan\s+be\s+seen\b", re.IGNORECASE)
SENT_RE = re.compile(r"[^.!?\n][^.!?]*[.!?]+")
THE_BACKREF_RE = re.compile(r"\bthe\s+([a-z]+)\b", re.IGNORECASE)

# 纯语法级推测名词短语过滤（零场景/测试词表依赖，保留 "appear happy" 等正常神态谓语）
NOUN_GUESS_HEDGE_RE = re.compile(
    r"\b(?:possibly|probably|perhaps|presumably|conceivably|"
    r"likely\s+(?:a|an|the)\b|,\s*likely\s+[a-z]+\b|"
    r"might\s+(?:be|have)\b|could\s+(?:be|have)\b|may\s+(?:be|have)\b|"
    r"looks?\s+like\s+(?:a|an|the)\b|resembl(?:es|ing)\s+(?:a|an|the)\b|"
    r"appear(?:s|ing)?\s+to\s+(?:be\s+(?:a|an|the|composed|made|covered|filled)\b|have\b)|"
    r"seem(?:s|ing)?\s+to\s+(?:be\s+(?:a|an|the|composed|made|covered|filled)\b|have\b))\b",
    re.IGNORECASE,
)
EPISTEMIC_HEDGE_RE = NOUN_GUESS_HEDGE_RE

_W: dict = {}
_LEM = WordNetLemmatizer()


GENERIC_HYPERNYM_HEADS = {"object", "item", "thing", "element", "entity", "figure", "shape", "form"}


def canon_noun_phrase(raw_s: str) -> tuple[bool, str]:
    """将名词短语还原为单数标准具象实体词干，过滤非具象抽象词头与泛指上位词。"""
    h = " ".join(str(raw_s or "").lower().split("[")[0].split())
    if not h or "|" in h:
        return False, ""
    toks = [t for t in WORD_TOK_RE.findall(h) if t not in {"a", "an", "the", "some", "several"}]
    if not toks:
        return False, ""
    toks[-1] = _LEM.lemmatize(toks[-1])
    cand_full = " ".join(toks)
    if cand_full in CONCRETE_COMPOUND_EXCEPTIONS:
        return True, norm_head(cand_full)
    if (
        cand_full in NON_CONCRETE_HEADS
        or toks[-1] in NON_CONCRETE_HEADS
        or cand_full in GENERIC_HYPERNYM_HEADS
        or toks[-1] in GENERIC_HYPERNYM_HEADS
    ):
        return False, ""
    # 若多词短语带有形容词修饰（如 "wooden bench"），优先归一至核心名词词头以对齐主实体键
    if len(toks) >= 2 and cand_full not in CONCRETE_COMPOUND_EXCEPTIONS:
        return True, norm_head(toks[-1])
    return True, norm_head(cand_full)


def extract_all_concrete_heads(f: dict) -> list[str]:
    """提取命题中涉及的全部具象实体词头（覆盖 entities[0]、entities[1..k]、align.object 及 identity 表语）。"""
    heads: list[str] = []
    conc0, h0 = is_concrete_entity(f)
    if conc0 and h0:
        ok0, ch0 = canon_noun_phrase(h0)
        if ok0 and ch0:
            heads.append(ch0)
    cat = f.get("category")
    ents = f.get("entities") or []
    for ent in ents[1:]:
        raw_e = ent.get("head") or ent.get("key") or ent.get("anchor") or ""
        ok_e, ch_e = canon_noun_phrase(raw_e)
        if ok_e and ch_e and ch_e not in heads:
            heads.append(ch_e)
    if cat in {"relation", "spatial", "identity"}:
        al_obj = (f.get("align") or {}).get("object") or (f.get("object") if cat == "identity" else "") or ""
        if al_obj and len(ents) >= (1 if cat == "identity" else 2):
            ok2, ch2 = canon_noun_phrase(al_obj)
            if ok2 and ch2 and ch2 not in heads:
                heads.append(ch2)
    return heads


def extract_foil_fsr(r: dict) -> float:
    """读取单图自反馈诱饵校准假支持率（综合全局与 object 类别假阳率）。"""
    cal = r.get("original_calibration") or {}
    base_fsr = float(cal.get("false_support_rate") or 0.0)
    obj_cal = (cal.get("by_category") or {}).get("object") or {}
    obj_fsr = float(obj_cal.get("false_support_rate") or 0.0)
    return max(base_fsr, obj_fsr)


def compile_calibrated_ledger_v3(facts: dict, gates: dict | None, foil_fsr: float = 0.0):
    """编译全槽位词干对齐 + Foil 假阳率与跨候选共识双向校准账本。"""
    rel_quotes: list[list[str]] = []
    obj_items: list[tuple[int, str, list[str], bool]] = []
    bad_count_quotes: list[list[str]] = []
    bad_other_ref_quotes: list[list[str]] = []
    gold_dv_heads: set[str] = set()
    cons_dv_heads: set[str] = set()

    req_min_p = max(0.68, min(0.76, 0.57 + 0.48 * foil_fsr))
    req_mean_p = max(0.72, min(0.80, 0.61 + 0.48 * foil_fsr))

    all_f = list(facts.values())
    if gates:
        for g in gates.values():
            all_f.extend((g.get("pending_checks") or {}).values())

    head_tot_counts: collections.Counter[str] = collections.Counter()
    head_fresh_s0: collections.Counter[str] = collections.Counter()
    for f in all_f:
        if f.get("polarity") == "positive" and f.get("category") in {"object", "identity"}:
            for ch in extract_all_concrete_heads(f):
                cnts = f.get("counts") or [0, 0, 0]
                fcnts = f.get("fresh_counts") or cnts
                head_tot_counts[ch] = max(head_tot_counts[ch], sum(cnts))
                head_fresh_s0[ch] = max(head_fresh_s0[ch], fcnts[0] if fcnts else 0)

    for f in all_f:
        if not verified(f):
            continue
        ss = effective_statuses(f)
        qs = [q["quote"].lower() for q in (f.get("quotes") or []) if q.get("quote")]
        if f.get("quote"):
            qs.append(f["quote"].lower())
        cat, pol, mod = f.get("category"), f.get("polarity"), f.get("modality")

        if "refuted" in ss and cat not in {"object", "identity"}:
            if qs:
                bad_other_ref_quotes.append(qs)
            continue
        if pol == "positive" and cat == "count":
            if ss.count("supported") == 0 and qs:
                bad_count_quotes.append(qs)
            continue
        if pol == "positive" and cat in {"spatial", "relation"} and ss == ["supported", "supported"]:
            if qs:
                rel_quotes.append(qs)
        if pol != "positive" or cat not in {"object", "identity"}:
            continue

        t_heads = extract_all_concrete_heads(f) if cat == "identity" else (
            [canon_noun_phrase(is_concrete_entity(f)[1])[1]] if is_concrete_entity(f)[0] else []
        )
        min_p, mean_p = fact_p_stats(f)
        for h in t_heads:
            if not h:
                continue
            tot_cnt = max(sum(f.get("counts") or [0, 0, 0]), head_tot_counts[h])
            fresh_c0 = max((f.get("fresh_counts") or f.get("counts") or [0])[0], head_fresh_s0[h])
            if "refuted" in ss:
                vtag = 3
            elif ss == ["supported", "supported"]:
                if mod == "uncertain" and min_p < 0.70:
                    vtag = 1
                elif tot_cnt <= 1 and not f.get("original_evidence_absent"):
                    need_min = max(0.72, req_min_p) if fresh_c0 == 0 else req_min_p
                    need_mean = max(0.76, req_mean_p) if fresh_c0 == 0 else req_mean_p
                    if min_p < need_min or mean_p < need_mean:
                        vtag = 1
                    else:
                        vtag = 0
                        gold_dv_heads.add(h)
                        cons_dv_heads.add(h)
                else:
                    vtag = 0
                    gold_dv_heads.add(h)
                    cons_dv_heads.add(h)
            elif ss.count("supported") == 1:
                if mod == "definite" and (fresh_c0 >= 2 or tot_cnt >= 4) and mean_p >= 0.66 and min_p >= 0.52:
                    vtag = 4
                    cons_dv_heads.add(h)
                else:
                    vtag = 1
            else:
                vtag = 2
            obj_items.append((vtag, h, qs, " " in h))

    return (rel_quotes, obj_items, bad_count_quotes, bad_other_ref_quotes), gold_dv_heads, cons_dv_heads


def eval_span_cal(text: str, compiled: tuple):
    """评估文本片段内的金牌双支持实体、高共识实体、关系数及各类泄漏信号。"""
    rel_quotes, obj_items, bad_count_quotes, bad_other_ref_quotes = compiled
    low = text.lower()
    raw_toks = WORD_TOK_RE.findall(low)
    wset = set(raw_toks) | {_LEM.lemmatize(w) for w in raw_toks}
    dv_heads: set[str] = set()
    cons_heads: set[str] = set()
    weak1_heads: set[str] = set()
    zero_heads: set[str] = set()
    ref_heads: set[str] = set()

    dv_rels = sum(1 for qs in rel_quotes if any(q in low for q in qs))
    bad_cnt = sum(1 for qs in bad_count_quotes if any(q in low for q in qs))
    bad_other_ref = sum(1 for qs in bad_other_ref_quotes if any(q in low for q in qs))

    for vtag, h, qs, is_mw in obj_items:
        matched = any(q in low for q in qs) or (((h in low) or ((h + "s") in low)) if is_mw else (h in wset))
        if not matched:
            continue
        if vtag == 0:
            dv_heads.add(h)
        elif vtag == 4:
            cons_heads.add(h)
        elif vtag == 1:
            weak1_heads.add(h)
        elif vtag == 2:
            zero_heads.add(h)
        else:
            ref_heads.add(h)

    cons_heads -= dv_heads
    weak1_heads -= dv_heads | cons_heads
    zero_heads -= dv_heads | cons_heads | weak1_heads
    ref_heads -= dv_heads
    return dv_heads, cons_heads, dv_rels, weak1_heads, zero_heads, ref_heads, bad_cnt, bad_other_ref


def check_fact_opt3(
    gf: dict | None,
    allowed_heads: set[str] | None = None,
    allow_1sup_cnt: bool = False,
) -> bool:
    """全槽位零盲区命题校验：
    - 若传入 allowed_heads（生产推荐）：要求全槽位具象实体均属于该图准入实体集（允许实体已核准但细微方位/动作为 unclear 的自然描述）；
    - 若未传入 allowed_heads（单命题独立校验模式）：拦截一切 0_sup 具象命题。
    """
    if gf is None or not verified(gf):
        return False
    ss = effective_statuses(gf)
    if "refuted" in ss:
        return False
    pol, cat, mod = gf.get("polarity"), gf.get("category"), gf.get("modality")
    all_conc = extract_all_concrete_heads(gf)

    if pol == "negative" and all_conc:
        return False

    if mod == "uncertain" and (all_conc or cat in {"object", "identity", "count"}):
        if ss != ["supported", "supported"] or fact_p_stats(gf)[0] < 0.70:
            return False
        if allowed_heads is not None and any(h not in allowed_heads for h in all_conc):
            return False

    if pol == "positive" and cat == "count":
        if ss == ["supported", "supported"] or (
            allow_1sup_cnt and ss.count("supported") == 1 and fact_p_stats(gf)[1] >= 0.63
        ):
            pass
        else:
            return False

    if allowed_heads is None:
        if all_conc and ss.count("supported") == 0:
            return False
        if pol == "positive" and cat in {"object", "identity"} and all_conc:
            return ss == ["supported", "supported"]
        return True

    if pol == "positive" and cat in {"object", "identity"} and all_conc:
        if any(h not in allowed_heads for h in all_conc) or ss.count("supported") == 0:
            return False
    elif pol == "positive" and all_conc:
        if any(h not in allowed_heads for h in all_conc):
            return False

    return True


def _trim_closing_fluff(cand_txt: str, comp: tuple, min_kept_words: int = 58) -> tuple[str, int, int]:
    """若 >=4 句候选的末句提供 0 新增核查具象实体且 0 空间关系，保留原始 \\n\\n 精确剥离末句空话。"""
    cur = cand_txt.strip()
    sents = list(SENT_RE.finditer(cur))
    if len(sents) >= 4:
        prev_end = sents[-2].end()
        pref = cur[:prev_end].strip()
        if len(pref.split()) >= min_kept_words:
            last_s = cur[sents[-1].start() :].strip()
            dv_p = eval_span_cal(pref, comp)[0]
            dv_l, _, rel_l = eval_span_cal(last_s, comp)[:3]
            if len(dv_l - dv_p) == 0 and rel_l == 0:
                return pref, 1, prev_end
    return cur, 0, len(cur)


def inspect_opt3_candidate(
    rc: dict,
    repaired_txt: str,
    orig_w: int,
    ledger_facts: dict,
    gate_info: dict | None,
    comp: tuple,
    gold_heads: set[str],
    cons_heads: set[str],
    aligner,
    vocab: list[str] | None,
    top_r: float = 0.25,
    allow_edit: bool = True,
    min_words_prefix: int = 58,
) -> tuple[int, dict, dict] | None:
    """评估单条候选的最优准入层级：
    1 = gold_full     (0 单支持、0 盲区、0 猜名词短语，全文/微修或仅剥离末句空话)
    2 = gold_rescued  (0 单支持、0 盲区，保留 \\n\\n 完整句前缀或指代安全单句切除)
    3 = cons_full     (至多 2 个跨样本强共识前景 1_sup，0 猜名词短语，全文/微修)
    4 = cons_prefix   (至多 2 个跨样本强共识前景 1_sup，保留 \\n\\n 完整句前缀)
    """
    if gate_info is not None and not gate_info.get("passed", False):
        return None
    p_checks = (gate_info or {}).get("pending_checks") or {}
    fact_keys = [(lf, aligner.canonical(lf, rc["id"])) for lf in rc.get("facts", [])]
    raw_txt = rc["text"].strip()

    def _verify_span(
        cand_txt: str,
        kept_ranges: list[tuple[int, int]],
        is_edited: bool,
        tier_mode: str,
        min_w: int,
        mode_lbl: str,
    ):
        trimmed_txt, fluff_cuts, prev_end = _trim_closing_fluff(cand_txt, comp, min_kept_words=max(58, min_w))
        if fluff_cuts > 0:
            if kept_ranges == [(0, len(cand_txt))]:
                kept_ranges = [(0, prev_end)]
            else:
                # 若原已在末尾截断过，基于新长度同步收缩最后一段范围
                last_a, last_b = kept_ranges[-1]
                shrink = len(cand_txt) - prev_end
                if last_b - shrink > last_a:
                    kept_ranges = kept_ranges[:-1] + [(last_a, last_b - shrink)]
            cand_txt = trimmed_txt

        eff_edit = is_edited or (fluff_cuts > 0)
        nw = len(cand_txt.split())
        if (
            nw < min_w
            or OR_SENT_RE.search(cand_txt)
            or NEG_SENT_RE.search(cand_txt)
            or NOUN_GUESS_HEDGE_RE.search(cand_txt)
        ):
            return None

        use_heads = gold_heads if tier_mode == "gold" else cons_heads
        allow_1cnt = tier_mode != "gold"
        low_c = cand_txt.lower()
        wset_c = set(WORD_TOK_RE.findall(low_c))
        ncache_c: list[str | None] = [None]
        kept_facts, kept_cids = [], set()

        for lf, key in fact_keys:
            gf = ledger_facts.get(key)
            in_kept = (not eff_edit) or _fast_fact_matches(lf, cand_txt, ncache_c, low_c, wset_c)
            if in_kept:
                if not check_fact_opt3(gf, use_heads, allow_1cnt):
                    return None
                if eff_edit:
                    kept_facts.append(copy.deepcopy(lf))
                kept_cids.add(key)
            elif not check_fact_opt3(gf, use_heads, allow_1cnt) and _fast_fact_matches(
                lf, cand_txt, ncache_c, low_c, wset_c
            ):
                return None

        for pf in p_checks.values():
            if not check_fact_opt3(pf, use_heads, allow_1cnt) and (
                (not eff_edit) or _fast_fact_matches(pf, cand_txt, ncache_c, low_c, wset_c)
            ):
                return None

        if not kept_cids:
            return None

        sdv, scons, srel, sweak1, szero, sref, sbcnt, sbref = eval_span_cal(cand_txt, comp)
        if sweak1 or szero or sref or sbcnt > 0 or sbref > 0:
            return None
        if tier_mode == "gold" and scons:
            return None
        if tier_mode != "gold" and len(scons) > 2:
            return None
        all_dv = sdv | scons
        if not all_dv:
            return None

        if not eff_edit:
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
            p2_novel = len(eval_span_cal(p1, comp)[0] - eval_span_cal(p0, comp)[0])

        raw_toks = WORD_TOK_RE.findall(low_c)
        rep_ratio = round(len(raw_toks) / max(1, len(set(raw_toks))), 3)

        meta = dict(
            w=nw,
            cut=max(0, int(orig_w) - nw),
            fluff_cuts=fluff_cuts,
            dv_div=round(effective_dv_diversity_score(all_dv), 4),
            dv_n=len(all_dv),
            dv_rels=int(srel),
            p2_novel=int(p2_novel),
            paras=cand_txt.count("\n\n") + 1,
            n_cons1=len(scons),
            n_1sup=len(scons),
            cnt_1sup=0,
            rep_ratio=rep_ratio,
            mode=(
                mode_lbl
                if is_edited
                else ("fluff_trim" if fluff_cuts > 0 else ("repaired" if cand_txt != raw_txt else "full"))
            ),
            needs_rescore=eff_edit,
            is_s0=(":s0:" in rc["id"]),
            has_cbs=bool(CBS_RE.search(cand_txt)),
        )
        return new_rc, meta

    full_vars = [raw_txt] + ([repaired_txt] if (repaired_txt and repaired_txt != raw_txt) else [])
    for ct in full_vars:
        if max(0, orig_w - len(ct.split())) <= 18:
            r1 = _verify_span(ct, [(0, len(ct))], False, "gold", 48, "full")
            if r1 is not None:
                return 1, r1[0], r1[1]

    if allow_edit:
        src_list = [repaired_txt, raw_txt] if (repaired_txt and repaired_txt != raw_txt) else [raw_txt]
        for src_t in src_list:
            sents = list(SENT_RE.finditer(src_t))
            if len(sents) >= 4:
                for dk in (1, 2):
                    if len(sents) - dk < 3:
                        continue
                    end_p = sents[-1 - dk].end()
                    pref = src_t[:end_p].strip()
                    if len(pref.split()) < min_words_prefix or max(0, orig_w - len(pref.split())) > 44:
                        continue
                    r2 = _verify_span(pref, [(0, end_p)], True, "gold", min_words_prefix, "prefix")
                    if r2 is not None:
                        return 2, r2[0], r2[1]

                for drop_i in range(len(sents) - 2, 0, -1):
                    dropped_s = src_t[sents[drop_i].start() : sents[drop_i].end()]
                    next_s = src_t[sents[drop_i + 1].start() : sents[drop_i + 1].end()]
                    if NON_OPENING_START_RE.match(next_s):
                        continue
                    d_words = {_LEM.lemmatize(w) for w in WORD_TOK_RE.findall(dropped_s.lower()) if len(w) > 2}
                    prev_txt = src_t[: sents[drop_i].start()]
                    p_words = {_LEM.lemmatize(w) for w in WORD_TOK_RE.findall(prev_txt.lower()) if len(w) > 2}
                    after_txt = src_t[sents[drop_i + 1].start() :]
                    the_refs = {_LEM.lemmatize(w) for w in THE_BACKREF_RE.findall(after_txt.lower())}
                    if the_refs & (d_words - p_words):
                        continue
                    sep = "\n\n" if "\n\n" in src_t[sents[drop_i - 1].end() : sents[drop_i + 1].start()] else " "
                    joined = prev_txt.rstrip() + sep + after_txt.lstrip()
                    if len(joined.split()) < min_words_prefix or max(0, orig_w - len(joined.split())) > 40:
                        continue
                    kr = [(0, sents[drop_i - 1].end()), (sents[drop_i + 1].start(), len(src_t))]
                    r2b = _verify_span(joined, kr, True, "gold", min_words_prefix, "coref_1drop")
                    if r2b is not None:
                        return 2, r2b[0], r2b[1]

    for ct in full_vars:
        if max(0, orig_w - len(ct.split())) <= 18:
            r3 = _verify_span(ct, [(0, len(ct))], False, "cons", 48, "cons_full")
            if r3 is not None:
                return 3, r3[0], r3[1]

    if allow_edit:
        src_list = [repaired_txt, raw_txt] if (repaired_txt and repaired_txt != raw_txt) else [raw_txt]
        cons_pref_min = max(54, min_words_prefix - 2)
        for src_t in src_list:
            sents = list(SENT_RE.finditer(src_t))
            if len(sents) >= 4:
                for dk in (1, 2):
                    if len(sents) - dk < 3:
                        continue
                    end_p = sents[-1 - dk].end()
                    pref = src_t[:end_p].strip()
                    if len(pref.split()) < cons_pref_min or max(0, orig_w - len(pref.split())) > 44:
                        continue
                    r4 = _verify_span(pref, [(0, end_p)], True, "cons", cons_pref_min, "cons_prefix")
                    if r4 is not None:
                        return 4, r4[0], r4[1]

    return None


def _init_worker(gen_cfg_path: str, spm_model_path: str):
    cfg = cfgmod.load(Path(gen_cfg_path))
    cfg["vendor"] = str(ROOT / "runtime/vendor")
    _W["cfg"] = cfg
    _W["aligner"] = make_aligner(cfg)
    _W["vocab"] = load_spm_pieces(Path(spm_model_path))
    _W["risk"] = read(ROOT / "models/results_v1_2.json")["risk_model"]
    _W["rej"] = read(ROOT / "models/hc_reject_models.json")


def _eval_record_opt3(r: dict, iid: str, cfg: dict, aligner, vocab: list[str], min_words_prefix: int = 58):
    facts = r["facts"]
    gates = r.get("naming_gates") or {}
    fsr = extract_foil_fsr(r)
    comp, gold_h, cons_h = compile_calibrated_ledger_v3(facts, gates, fsr)
    old_im = r["original_features"]
    rep_im = r.get("repaired_features") or old_im
    old_by_id = {c["id"]: c for c in old_im["candidates"]}
    rep_by_id = {c["id"]: c["text"].strip() for c in rep_im["candidates"]}

    cand_results: dict[str, tuple[int, dict, dict]] = {}
    for rc in r["candidates"]:
        cid = rc["id"]
        ev = inspect_opt3_candidate(
            rc=rc,
            repaired_txt=rep_by_id.get(cid, ""),
            orig_w=old_by_id[cid]["words"],
            ledger_facts=facts,
            gate_info=gates.get(cid),
            comp=comp,
            gold_heads=gold_h,
            cons_heads=cons_h,
            aligner=aligner,
            vocab=vocab,
            top_r=cfg["fuzzy"]["top_r"],
            allow_edit=False,
            min_words_prefix=min_words_prefix,
        )
        if ev is not None:
            cand_results[cid] = ev

    t1_txts = {v[1]["text"].strip() for v in cand_results.values() if v[0] == 1}
    if len(t1_txts) < 2:
        for rc in r["candidates"]:
            cid = rc["id"]
            if cid in cand_results and cand_results[cid][0] == 1:
                continue
            ev = inspect_opt3_candidate(
                rc=rc,
                repaired_txt=rep_by_id.get(cid, ""),
                orig_w=old_by_id[cid]["words"],
                ledger_facts=facts,
                gate_info=gates.get(cid),
                comp=comp,
                gold_heads=gold_h,
                cons_heads=cons_h,
                aligner=aligner,
                vocab=vocab,
                top_r=cfg["fuzzy"]["top_r"],
                allow_edit=True,
                min_words_prefix=min_words_prefix,
            )
            if ev is not None:
                cand_results[cid] = ev

    # 若在 min_words_prefix=58 下整图无候选，温和回退至 52 词再抢救一次纯金前缀
    if not cand_results and min_words_prefix > 52:
        for rc in r["candidates"]:
            cid = rc["id"]
            ev = inspect_opt3_candidate(
                rc=rc,
                repaired_txt=rep_by_id.get(cid, ""),
                orig_w=old_by_id[cid]["words"],
                ledger_facts=facts,
                gate_info=gates.get(cid),
                comp=comp,
                gold_heads=gold_h,
                cons_heads=cons_h,
                aligner=aligner,
                vocab=vocab,
                top_r=cfg["fuzzy"]["top_r"],
                allow_edit=True,
                min_words_prefix=52,
            )
            if ev is not None:
                cand_results[cid] = ev

    if not cand_results:
        return None

    t1_n = len({v[1]["text"].strip() for v in cand_results.values() if v[0] == 1})
    t12_n = len({v[1]["text"].strip() for v in cand_results.values() if v[0] <= 2})
    t3_n = len({v[1]["text"].strip() for v in cand_results.values() if v[0] <= 3})

    if t1_n >= 2:
        max_tier = 1
        tier_tag = "tier1_gold_full"
    elif t12_n >= 2:
        max_tier = 2
        tier_tag = "tier2_gold_rescued"
    elif t3_n >= 1:
        max_tier = 3 if t3_n >= 2 else 4
        tier_tag = "tier3_cons_full" if max_tier == 3 else "tier4_cons_mixed"
    else:
        max_tier = 4
        tier_tag = "tier4_cons_prefix"

    selected_cids = {cid for cid, v in cand_results.items() if v[0] <= max_tier}
    metas = {cid: cand_results[cid][2] for cid in selected_cids}
    needs_rescore = any(metas[cid]["needs_rescore"] for cid in selected_cids)

    if not needs_rescore and r.get("repaired_features") is not None:
        rep_cands_map = {c["id"]: c for c in rep_im["candidates"]}
        merged_cands = []
        for rc in r["candidates"]:
            cid_k = rc["id"]
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
        new_allf = {iid + "|" + k: v for k, v in extra_feature_map(facts, merged_cands).items()}
    else:
        ncs = [cand_results[rc["id"]][1] if rc["id"] in selected_cids else copy.deepcopy(rc) for rc in r["candidates"]]
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

    best_key = max(
        (metas[c["id"]]["n_cons1"] == 0, metas[c["id"]]["dv_div"], metas[c["id"]]["p2_novel"]) for c in allowed
    )
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
    (
        chunk_items,
        src_data_dir,
        guided_map,
        wp_a,
        wp_b,
        s0_boost,
        p1_b,
        qv_bonus,
        min_words_prefix,
        compute_hashes,
    ) = args_tuple
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
        pack = _eval_record_opt3(base_r, iid, cfg, aligner, vocab, min_words_prefix=min_words_prefix)

        if iid in guided_map and (pack is None or pack["tier_tag"] != "tier1_gold_full"):
            g_path = Path(guided_map[iid])
            g_r = read(g_path)
            if str(g_r["record"]["id"]) == iid and g_r["image_sha256"] == entry["image_sha256"]:
                g_pack = _eval_record_opt3(g_r, iid, cfg, aligner, vocab, min_words_prefix=min_words_prefix)
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
            chunk_audits.append(
                dict(id=iid, tier="none", pairs=0, eligible=0, skip="no_clean_opt3_candidate", construction=[])
            )
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

        old_allf = {iid + "|" + k: v for k, v in extra_feature_map(facts, old_im["candidates"]).items()}
        cons, cons2 = consensus_from_original(iid, new_im["candidates"], facts, old_im["candidates"])

        fallback_neg_cands = []
        fallback_old_allf = {}
        if use_path != base_path:
            b_old = base_r["original_features"]
            fallback_old_allf = {
                iid + "|" + k: v for k, v in extra_feature_map(base_r["facts"], b_old["candidates"]).items()
            }
            fallback_neg_cands = b_old["candidates"]

        def _risk(c):
            z = linear(risk_model, risk_features(c, cons[(iid, c["id"])], cons2[(iid, c["id"])]))
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

        risks = {c["id"]: _risk(c) for c in allowed}
        base = max(allowed, key=anchor_score)
        gold_only = [c for c in allowed if metas[c["id"]]["n_cons1"] == 0]
        base_pool = gold_only or allowed
        elig_a = [c for c in base_pool if risks[c["id"]] <= risks[base["id"]] + 0.03] or base_pool
        pool_a = [c for c in elig_a if metas[c["id"]]["mode"] not in {"prefix", "cons_prefix"}] or elig_a

        def _sa(c):
            mt = metas[c["id"]]
            s0_b = 0.024 if mt["is_s0"] else (-0.010 if mt["has_cbs"] else 0.0)
            p2_add = 0.068 * min(3, mt["p2_novel"]) if (mt["paras"] >= 2 and mt["p2_novel"] >= 1) else 0.0
            density = mt["dv_div"] / max(50.0, float(c["words"]))
            return 0.50 * mix(c) + 0.50 * qv(c) + 0.034 * min(14.0, mt["dv_div"]) + 0.14 * density + p2_add + s0_b

        def _sb(c):
            mt = metas[c["id"]]
            ps = (0.26 * mix(c) + 0.74 * qv(c)) * (
                0.05 + 0.95 * (c["p"] or 0.0) * (1.0 - c["unsupported_primary_fraction"])
            ) ** 2
            p2_mult = 0.105 * min(4, mt["p2_novel"]) if (mt["paras"] >= 2 and mt["p2_novel"] >= 1) else 0.0
            return ps * (1.0 + 0.046 * min(15.0, mt["dv_div"]) + p2_mult)

        ch_a = max(pool_a, key=lambda c: (round(_sa(c), 12), -risks[c["id"]]))
        rem_b = [c for c in allowed if c["text"].strip() != ch_a["text"].strip()]
        ch_b = max(rem_b, key=lambda c: (round(_sb(c), 12), -risks[c["id"]])) if rem_b else ch_a
        sel = {"v1.2": ch_a, "product": ch_b}

        img_pairs = []
        c_audit = []
        seen_pair_keys: set[tuple[str, str]] = set()
        used_rejs: set[str] = set()

        for role, other in [("v1.2", "product"), ("product", "v1.2")]:
            ch = sel[role]
            if role == "product" and ch["text"].strip() == sel["v1.2"]["text"].strip():
                continue
            excl = {ch["id"], sel[other]["id"]} | used_rejs
            pc = float(ch["p"] or 0.0)
            mc = mix(ch)
            qvc = qv(ch)
            wc = float(ch["words"])
            ch_strip = ch["text"].strip()
            ch_meta = metas[ch["id"]]
            fc = reject_features(ch, new_allf[iid + "|" + ch["id"]])

            def _pick_rej(cand_list, fmap):
                # 优先池：严格正向/微容忍覆盖单调守卫（杜绝 56.4% 的 delta_qv < 0 倒挂）
                vh = [
                    rc
                    for rc in cand_list
                    if rc["id"] not in excl
                    and len(rc["errors"]) >= 1
                    and not rc["text"].strip().startswith(ch_strip)
                    and not ch_strip.startswith(rc["text"].strip())
                    and (pc - float(rc["p"] or 0.0)) > 0.015
                    and (mc - mix(rc)) >= -0.028
                    and (qvc - qv(rc)) >= -0.018
                    and purity_eval(ch, rc)["passed"]
                ]
                if not vh:
                    vh = [
                        rc
                        for rc in cand_list
                        if rc["id"] not in excl
                        and len(rc["errors"]) >= 1
                        and not rc["text"].strip().startswith(ch_strip)
                        and not ch_strip.startswith(rc["text"].strip())
                        and (pc - float(rc["p"] or 0.0)) > 0.025
                        and (mc - mix(rc)) >= -0.055
                        and (qvc - qv(rc)) >= -0.055
                        and purity_eval(ch, rc)["passed"]
                    ]
                if not vh:
                    return None

                if role == "v1.2":
                    rpool = [rc for rc in vh if len(rc["errors"]) >= 2] or vh

                    def _rsa(rc):
                        fr = reject_features(rc, fmap[iid + "|" + rc["id"]])
                        bs = linear(reject_models[role]["model"], _concat_diff(fr, fc))
                        w_diff = (rc["words"] - wc) / max(1.0, wc)
                        return bs - wp_a * max(0.0, w_diff) + qv_bonus * (qvc - qv(rc))

                    return max(rpool, key=_rsa)

                def _rsb(rc):
                    fr = reject_features(rc, fmap[iid + "|" + rc["id"]])
                    bs = linear(reject_models[role]["model"], _concat_diff(fr, fc))
                    w_diff = (rc["words"] - wc) / max(1.0, wc)
                    s0_b = s0_boost if ":s0:" in rc["id"] else 0.0
                    # 仅当正例具有真实新增双核查实体的第二段时给予温和分段引导
                    p1_nudge = (
                        p1_b if ("\n\n" in ch_strip and ch_meta["p2_novel"] >= 1 and "\n\n" not in rc["text"]) else 0.0
                    )
                    return bs + s0_b - wp_b * max(0.0, w_diff) + qv_bonus * (qvc - qv(rc)) + p1_nudge

                return max(vh, key=_rsb)

            rej = _pick_rej(old_im["candidates"], old_allf)
            if rej is None and fallback_neg_cands:
                rej = _pick_rej(fallback_neg_cands, fallback_old_allf)

            aud = dict(
                role=role,
                chosen_id=ch["id"],
                rejected_id=None if rej is None else rej["id"],
                chosen_opt3_meta=ch_meta,
            )
            if rej is None:
                aud["reason"] = "no_eligible_hard_reject"
            else:
                pur = purity_eval(ch, rej)
                aud.update(purity=pur, reason=pur["reason"])
                pkey = (ch_strip, rej["text"].strip())
                if pur["passed"] and ch_strip != rej["text"].strip() and pkey not in seen_pair_keys:
                    seen_pair_keys.add(pkey)
                    used_rejs.add(rej["id"])
                    orig_text = old_by_id.get(ch["id"], ch)["text"]
                    opt_meta = dict(
                        ch_meta,
                        delta_p=round(pc - float(rej["p"] or 0.0), 6),
                        delta_mix=round(mc - mix(rej), 6),
                        delta_qv=round(qvc - qv(rej), 6),
                    )
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
                            opt_meta=opt_meta,
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
    ap.add_argument("--out-run", type=str, default="full10000_v3_opt3")
    ap.add_argument("--pair-name", type=str, default="E3_V3_OPT3.json")
    ap.add_argument("--run-group", type=str, default="E3_V3_OPT3")
    ap.add_argument("--wp-a", type=float, default=0.22)
    ap.add_argument("--wp-b", type=float, default=2.15)
    ap.add_argument("--s0-boost", type=float, default=0.65)
    ap.add_argument("--p1-b", type=float, default=0.09)
    ap.add_argument("--qv-bonus", type=float, default=0.65)
    ap.add_argument("--min-words-prefix", type=int, default=58)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run-step", type=int, default=0)
    a = ap.parse_args()

    R = a.v3_root.resolve()
    manifest_path = R / "manifests/source_index.json"
    heldout_path = R / "evaluation/heldout_image_manifest_all.json"
    pixel_path = R / "checks/training_image_pixel_hashes.json"
    src_data_dir = R / "data" / a.src_run
    guided_dir = R / "data" / a.guided_run

    gen_cfg_path = a.v2_root.resolve() / "evaluation/generation.json"
    spm_candidates = [
        R / "training/models/E3_V3_OPT2/tokenizer.model",
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
    # 分块粒度 ~250 图，便于流式打印实时进度并避免长时静默
    chunk_size = max(1, min(250, math.ceil(len(flat_items) / n_workers)))
    compute_hashes = a.dry_run_step == 0
    chunks = [
        (
            flat_items[i : i + chunk_size],
            str(src_data_dir),
            guided_map,
            a.wp_a,
            a.wp_b,
            a.s0_boost,
            a.p1_b,
            a.qv_bonus,
            a.min_words_prefix,
            compute_hashes,
        )
        for i in range(0, len(flat_items), chunk_size)
    ]

    all_pairs, all_audits, all_sources = [], [], {}
    stats = collections.Counter()
    done_imgs = 0
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
            done_imgs += st.get("source_images", 0)
            print(
                f"[OPT3_BUILD] {done_imgs}/{len(flat_items)} images | raw_pairs={len(all_pairs)} | none={stats.get('pool_none', 0)}",
                flush=True,
            )

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
    dqvs = [r["opt_meta"]["delta_qv"] for r in clean_rows]
    dmixs = [r["opt_meta"]["delta_mix"] for r in clean_rows]

    wgt_c_gt_r = (
        sum(float(r["pair_weight"]) for r in clean_rows if len(r["chosen"].split()) > len(r["rejected"].split()))
        / w_tot
        * 100.0
    )
    wgt_c2 = sum(float(r["pair_weight"]) for r in clean_rows if "\n\n" in r["chosen"]) / w_tot * 100.0
    wgt_r2 = sum(float(r["pair_weight"]) for r in clean_rows if "\n\n" in r["rejected"]) / w_tot * 100.0

    c_s0 = sum(1 for r in clean_rows if ":s0:" in r["chosen_id"])
    r_s0 = sum(1 for r in clean_rows if ":s0:" in r["rejected_id"])
    n_cons1_rows = sum(1 for r in clean_rows if r["opt_meta"]["n_cons1"] > 0)
    hedge_rows = sum(1 for r in clean_rows if NOUN_GUESS_HEDGE_RE.search(r["chosen"]))
    uniq_chosen_pct = len({(r["image_id"], r["chosen"].strip()) for r in clean_rows}) / len(clean_rows) * 100.0

    stats["training_pairs"] = len(clean_rows)
    stats["training_images"] = len(img_w_sums)
    stats["chosen_or"] = sum(bool(OR_SENT_RE.search(r["chosen"])) for r in clean_rows)
    stats["chosen_neg"] = sum(bool(NEG_SENT_RE.search(r["chosen"])) for r in clean_rows)
    stats["chosen_hedge"] = hedge_rows
    stats["chosen_G_errors"] = sum(r["chosen_G_errors"] for r in clean_rows)
    stats["rejected_G_errors_total"] = sum(r["rejected_G_errors"] for r in clean_rows)

    if (
        stats["chosen_or"] != 0
        or stats["chosen_neg"] != 0
        or stats["chosen_hedge"] != 0
        or stats["chosen_G_errors"] != 0
    ):
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
        chosen_hedge=stats["chosen_hedge"],
        avg_cut_words=round(statistics.mean(cuts), 3),
        chosen_words_avg=round(statistics.mean(cw_list), 2),
        rejected_words_avg=round(statistics.mean(rw_list), 2),
        word_diff_avg=round(statistics.mean(cw_list) - statistics.mean(rw_list), 2),
        wgt_C_gt_R_pct=round(wgt_c_gt_r, 2),
        chosen_2para_pct=round(wgt_c2, 2),
        rejected_2para_pct=round(wgt_r2, 2),
        wgt_c2_minus_r2_pct=round(wgt_c2 - wgt_r2, 2),
        chosen_s0_pct=round(c_s0 / len(clean_rows) * 100.0, 2),
        rejected_s0_pct=round(r_s0 / len(clean_rows) * 100.0, 2),
        dv_div_avg=round(statistics.mean(dvs), 3),
        p2_novel_avg=round(statistics.mean(p2s), 3),
        delta_qv_avg=round(statistics.mean(dqvs), 4),
        delta_mix_avg=round(statistics.mean(dmixs), 4),
        cons1_rescue_pct=round(n_cons1_rows / len(clean_rows) * 100.0, 2),
        weak_1sup_concrete_pct=0.0,
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
    opt3_cfg = dict(
        base_train_cfg,
        pairs=str(pair_file),
        run_group=a.run_group,
        checkpoint_policy="steps_150_plus_final",
        save_steps=150,
    )
    cfg_out = R / f"training_handoff/a100_training_{a.run_group.lower()}.json"
    write(cfg_out, opt3_cfg)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
