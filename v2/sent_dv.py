"""E3-V3Z 句级双核查零泄漏外科裁剪模块（Sentence-Level Double-Verification Surgical Pruning）。

设计目标（完全盲标，严格零测试集词表或标注参与）：
1. 根治“实例级幻觉率低但篇级 Hal/CHAIRs/Resp 高”的二项式分散泄漏：
   - 任何单票支持（['supported', 'unclear']）或零票支持（['unclear', 'unclear']）的具象物理实体/身份命题所在句一律切除；
   - 对仅被 <=1 条候选提及的孤立新实体，额外施加双核查置信度下限（min_p >= 0.70, mean_p >= 0.74），拦截 LLaVA 视觉问答共现 Yes 偏置假阳性；
   - 跨确定/不确定双模态统一拦截任何含 'refuted' 的命题、任何未消解 'or'、任何显式否定缺失结构及推测性模糊措辞（appear/seem/possibly/likely/might/...）。
2. 保持篇章连贯与因果模糊熵精确性：
   - 要求首句（Sentence 0）无污染以稳固主语锚定；
   - 若中间句被切除，通过句首悬垂代词检测与已删句定冠词名词回指守卫（has_broken_backref）阻断断裂指代；
   - 对连续前缀句直接从已保存的因果 teacher-forcing token_ids 与 entropies 无损精确重算模糊熵 u。
"""
from __future__ import annotations

from copy import deepcopy
import re

from rlsfv.facts import norm
from rlsfv.scoring import aggregate_fuzzy, effective_statuses, verified
from .ledger_guard import is_concrete_entity

OR_SENT_RE = re.compile(r"\bor\b", re.IGNORECASE)
NEG_SENT_RE = re.compile(
    r"\b(?:with\s+no|without\s+(?:any\s+)?|no\s+visible|not\s+visible|cannot\s+be\s+seen|"
    r"void\s+of|free\s+of|neither\s+\w+\s+nor|devoid\s+of)\b",
    re.IGNORECASE,
)
SPEC_SENT_RE = re.compile(
    r"\b(?:appear(?:s|ing)?\b|seem(?:s|ing|ingly)?\b|possibly|probably|likely|perhaps|"
    r"might\b|could\b|may\s+be\b|suggest(?:s|ing)|indicat(?:es|ing)|implying|implies)\b",
    re.IGNORECASE,
)
DANGLING_START_RE = re.compile(
    r"^\s*(?:This|That|These|Those|He|She|It|They|His|Her|Their|Its|Both|Such|Another|"
    r"The\s+other|The\s+same|Beside\s+(?:it|them|him|her)|Next\s+to\s+(?:it|them|him|her)|"
    r"On\s+top\s+of\s+it|Around\s+(?:it|them))\b",
    re.IGNORECASE,
)


def fact_p_stats(gf: dict) -> tuple[float, float]:
    """提取双模板视觉核查的有效最小与平均真值概率。"""
    pts = [
        float(ch["p_true"])
        for ch in gf.get("checks", [])
        if isinstance(ch.get("p_true"), (int, float)) and not ch.get("low_mass") and ch.get("valid", True)
    ]
    if len(pts) == 2:
        return min(pts), 0.5 * (pts[0] + pts[1])
    return 0.0, 0.0


def check_fact_strict_dv(gf: dict | None) -> bool:
    """判定单条命题是否满足句级双核查零泄漏标准。"""
    if gf is None or not verified(gf):
        return False
    ss = effective_statuses(gf)
    if "refuted" in ss:
        return False
    cat = gf.get("category")
    pol = gf.get("polarity")
    if pol == "negative" and cat in {"object", "identity"}:
        conc, _ = is_concrete_entity(gf)
        if conc:
            return False
    if pol == "positive" and cat in {"object", "identity"}:
        conc, _ = is_concrete_entity(gf)
        if conc:
            if ss != ["supported", "supported"]:
                return False
            tot_cnt = sum(gf.get("counts") or [0, 0, 0])
            min_p, mean_p = fact_p_stats(gf)
            # 孤立单样本新实体须具备明确高置信视觉支持，拦截 0.65 临界共现偏置幻觉
            if tot_cnt <= 1 and (min_p < 0.70 or mean_p < 0.74):
                return False
    if pol == "positive" and cat == "count":
        if ss.count("supported") == 0:
            return False
    return True


def map_fact_sents(lf: dict, sents: list) -> list[int]:
    """将候选局部命题映射至所属句子下标列表。"""
    si = lf.get("sent_index")
    if isinstance(si, int) and 0 <= si < len(sents):
        q = (lf.get("quote") or "").strip()
        if not q or q in sents[si].text:
            return [si]
    q = (lf.get("quote") or "").strip()
    if q:
        matched = [idx for idx, s in enumerate(sents) if q in s.text]
        if matched:
            return matched
        qn = norm(q)
        matched_n = [idx for idx, s in enumerate(sents) if qn and qn in norm(s.text)]
        if matched_n:
            return matched_n
    return []


def sent_nouns(sent_span) -> set[str]:
    """提取句内实词名词词元，用于跨句回指完备性检查。"""
    return {t.lemma_.lower() for t in sent_span if t.pos_ in {"NOUN", "PROPN"} and len(t.lemma_) > 1}


def has_broken_backref(sent_span, dropped_prev: bool, kept_nouns: set[str], dropped_nouns: set[str]) -> bool:
    """检测候选保留句是否存在指向已被切除前句的悬垂代词或定冠词回指。"""
    st = sent_span.text
    if dropped_prev and DANGLING_START_RE.match(st):
        return True
    toks = list(sent_span)
    for idx, t in enumerate(toks):
        if t.lower_ == "the" and idx + 1 < len(toks):
            for j in range(idx + 1, min(len(toks), idx + 4)):
                if toks[j].pos_ in {"NOUN", "PROPN"}:
                    lem = toks[j].lemma_.lower()
                    if lem in dropped_nouns and lem not in kept_nouns:
                        return True
                    break
    return False


def exact_prefix_u(rc: dict, prefix_text: str, tokenizer, pos_parser, fuzzy_cfg: dict) -> tuple[float | None, bool]:
    """利用自回归模型因果下三角掩码性质，对精确 token 前缀无损计算模糊熵 u。"""
    fuz = rc.get("fuzzy") or {}
    orig_tids = fuz.get("token_ids")
    orig_ents = fuz.get("entropies")
    if not orig_tids or not orig_ents:
        return None, False
    ptids = tokenizer.encode(prefix_text, add_special_tokens=False)
    k = len(ptids)
    if k == 0 or k > len(orig_tids) or orig_tids[:k] != ptids:
        return None, False
    pieces = tokenizer.convert_ids_to_tokens(ptids)
    pieces = [t.replace("▁", " ").replace("<0x0A>", "\n") for t in pieces]
    pieces = ["" if t.startswith("<") and t.endswith(">") else t for t in pieces]
    spans = pos_parser.pos_spans("".join(pieces), fuzzy_cfg["pos_tags"])
    mask, offset = [], 0
    for piece in pieces:
        end = offset + len(piece)
        mask.append(any(a < end and b > offset for a, b in spans) if end > offset else False)
        offset = end
    try:
        return aggregate_fuzzy(orig_ents[:k], mask, fuzzy_cfg["top_r"]), True
    except ValueError:
        return None, False


def subset_token_u(rc: dict, sents: list, kept_indices: list[int], tokenizer, fuzzy_cfg: dict) -> tuple[float, bool]:
    """对保留的非连续安全句集提取对应实词跨度模糊熵。"""
    fuz = rc.get("fuzzy") or {}
    orig_tids = fuz.get("token_ids")
    orig_ents = fuz.get("entropies")
    if not orig_tids or not orig_ents:
        return float(fuz.get("u") or 0.0), False
    pieces = tokenizer.convert_ids_to_tokens(orig_tids)
    pieces = [t.replace("▁", " ").replace("<0x0A>", "\n") for t in pieces]
    pieces = ["" if t.startswith("<") and t.endswith(">") else t for t in pieces]
    kept_ranges = [(sents[i].start_char, sents[i].end_char) for i in kept_indices]
    joined = "".join(pieces)
    lead = len(joined) - len(joined.lstrip())
    sel_ents, sel_mask = [], []
    orig_mask = fuz.get("content_mask") or [True] * len(orig_ents)
    offset = 0
    for idx, piece in enumerate(pieces):
        end = offset + len(piece)
        c0, c1 = max(0, offset - lead), max(0, end - lead)
        if end > offset and any(c0 < r1 and c1 > r0 for r0, r1 in kept_ranges):
            sel_ents.append(orig_ents[idx])
            sel_mask.append(orig_mask[idx])
        offset = end
    try:
        return aggregate_fuzzy(sel_ents, sel_mask, fuzzy_cfg["top_r"]), True
    except ValueError:
        return float(fuz.get("u") or 0.0), False


def build_sent_dv_candidate(
    rc: dict,
    ledger_facts: dict,
    gate_info: dict | None,
    nlp,
    aligner,
    tokenizer,
    pos_parser,
    fuzzy_cfg: dict,
    min_words: int = 24,
) -> tuple[dict, bool, str]:
    """对单条候选执行句级双核查零泄漏外科裁剪，返回 (新候选字典, 是否满足零泄漏准入, 模式标签)。"""
    txt = rc["text"].strip()
    doc = nlp(txt)
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return deepcopy(rc), False, "empty"

    n_s = len(sents)
    sent_bad = [False] * n_s
    for i, s in enumerate(sents):
        st = s.text
        if OR_SENT_RE.search(st) or NEG_SENT_RE.search(st) or SPEC_SENT_RE.search(st):
            sent_bad[i] = True

    p_checks = (gate_info or {}).get("pending_checks") or {}
    for pf in p_checks.values():
        if not check_fact_strict_dv(pf):
            s_idxs = map_fact_sents(pf, sents)
            if s_idxs:
                for idx in s_idxs:
                    sent_bad[idx] = True
            else:
                return deepcopy(rc), False, "pending_bad_unmapped"

    fact_sidxs = []
    for lf in rc.get("facts", []):
        key = aligner.canonical(lf, rc["id"])
        gf = ledger_facts.get(key)
        s_idxs = map_fact_sents(lf, sents)
        fact_sidxs.append((lf, key, s_idxs))
        if not check_fact_strict_dv(gf):
            if s_idxs:
                for idx in s_idxs:
                    sent_bad[idx] = True
            else:
                return deepcopy(rc), False, "fact_bad_unmapped"

    if not any(sent_bad):
        return deepcopy(rc), True, "full_pure"

    if sent_bad[0]:
        return deepcopy(rc), False, "s0_bad"

    k_pref = 0
    while k_pref < n_s and not sent_bad[k_pref]:
        k_pref += 1

    kept_sub = list(range(k_pref))
    kept_nouns = set().union(*(sent_nouns(sents[i]) for i in kept_sub))
    dropped_nouns: set[str] = set()
    dropped_prev = False
    for i in range(k_pref, n_s):
        if sent_bad[i]:
            dropped_nouns |= sent_nouns(sents[i])
            dropped_prev = True
            continue
        if has_broken_backref(sents[i], dropped_prev, kept_nouns, dropped_nouns):
            dropped_nouns |= sent_nouns(sents[i])
            dropped_prev = True
            continue
        kept_sub.append(i)
        kept_nouns |= sent_nouns(sents[i])
        dropped_prev = False

    if kept_sub == list(range(k_pref)):
        new_txt = txt[: sents[k_pref - 1].end_char].strip()
        chosen_tag = "prefix"
    else:
        new_txt = " ".join(sents[i].text.strip() for i in kept_sub)
        chosen_tag = "subset"

    if len(new_txt.split()) < min_words:
        return deepcopy(rc), False, "too_short"

    kept_set = set(kept_sub)
    kept_facts, kept_cids = [], set()
    for lf, key, s_idxs in fact_sidxs:
        q = (lf.get("quote") or "").strip()
        in_kept = any(
            idx in kept_set and (not q or q in sents[idx].text or norm(q) in norm(sents[idx].text))
            for idx in s_idxs
        )
        if in_kept:
            kept_facts.append(deepcopy(lf))
            kept_cids.add(key)

    if not kept_cids:
        return deepcopy(rc), False, "empty_claims"

    if chosen_tag == "prefix":
        u_val, ok_u = exact_prefix_u(rc, new_txt, tokenizer, pos_parser, fuzzy_cfg)
    else:
        u_val, ok_u = subset_token_u(rc, sents, kept_sub, tokenizer, fuzzy_cfg)
    if not ok_u:
        return deepcopy(rc), False, "u_fail"

    new_rc = deepcopy(rc)
    new_rc["text"] = new_txt
    new_rc["facts"] = kept_facts
    new_rc["claim_ids"] = sorted(kept_cids)
    new_rc["pending_queries"] = []
    new_rc["fuzzy"] = dict(rc["fuzzy"], u=u_val)
    new_rc["repair"] = dict(
        policy="v3z_sent_dv",
        mode=chosen_tag,
        original_text=rc["text"],
        kept_sents=kept_sub,
        orig_sents=n_s,
    )
    return new_rc, True, chosen_tag
