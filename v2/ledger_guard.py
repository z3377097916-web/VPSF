"""E3-V3 证据账本跨度保护与具象实体边缘风险门控模块（完全盲标，不访问任何评测集词表或真值）。"""
from __future__ import annotations

import re
from rlsfv import lexicon as L
from rlsfv.scoring import effective_statuses
from .cover_budget import noun_heads, newly_lost_heads

OR_WORD_RE = re.compile(r"\bor\b", re.IGNORECASE)
NEG_ABSENCE_RE = re.compile(
    r"\b(?:with\s+no|without\s+(?:any\s+)?|no\s+visible|void\s+of|free\s+of|neither\s+\w+\s+nor)\b",
    re.IGNORECASE,
)

# 基于 rlsfv.lexicon 的盲标抽象/元话语/方位/量词名词集合（不纳入具象物体零支持误杀范围）
NON_CONCRETE_HEADS = (
    set(L.META_NOUNS)
    | set(L.POSITION_NOUNS)
    | set(L.ABSTRACT)
    | set(L.QUANT_NOUNS)
    | set(L.STYLE_NOUNS)
    | set(L.TIME)
    | set(L.PART_NOUNS)
    | set(L.SUBJECTIVE_OBJECTS)
    | {
        "viewer", "majority", "direction", "air", "abundance", "backdrop", "mix",
        "combination", "impression", "safety", "support", "overall", "importance",
        "beginning", "turn", "point", "display", "space", "light", "lighting",
        "shadow", "shadows", "reflection", "perspective", "angle", "position",
        "state", "condition", "quality", "texture", "pattern", "tone", "hue", "shade",
        "feel", "atmosphere", "sense", "presence", "pairing", "bit", "demeanor",
        "progress", "disrepair", "collision", "incident", "issue", "attention",
        "intrigue", "curiosity", "purpose", "norm", "result", "competition",
        "excitement", "moment", "activity", "event", "situation", "occasion",
    }
)

# 复合具象名词保护白名单（避免因末词落在方位/泛化词而漏检）
CONCRETE_COMPOUND_EXCEPTIONS = {
    "dining table", "coffee table", "picnic table", "side table", "traffic light",
    "street light", "tail light", "brake light", "fire hydrant", "stop sign",
    "street sign", "road sign", "parking meter", "sports ball", "tennis ball",
    "soccer ball", "baseball bat", "baseball glove", "tennis racket", "wine glass",
    "hot dog", "potted plant", "cell phone", "teddy bear", "hair drier",
    "cutting board", "bus stop", "train station",
}


def canonical_entity_head(fact: dict) -> str:
    """从原子事实提取归一化实体主词。"""
    ents = fact.get("entities") or []
    raw = (
        (ents[0].get("head") or ents[0].get("key") or fact.get("subject") or "")
        if ents
        else (fact.get("subject") or "")
    )
    head = " ".join(str(raw).lower().split("[")[0].split())
    return head


def is_concrete_entity(fact: dict) -> tuple[bool, str]:
    """判定对象/身份事实是否为具象物理实体（过滤元话语与抽象概念抽取噪声）。"""
    head = canonical_entity_head(fact)
    if not head:
        return False, ""
    if head in CONCRETE_COMPOUND_EXCEPTIONS:
        return True, head
    if head in NON_CONCRETE_HEADS:
        return False, head
    last_tok = head.split()[-1]
    if last_tok in NON_CONCRETE_HEADS:
        return False, head
    return True, head


def _head_lemmas(head: str) -> set[str]:
    if not head:
        return set()
    parts = [p for p in re.split(r"\s+", head.strip()) if p]
    out = {head}
    if parts:
        out.add(parts[-1])
    return out


def build_ledger_span_guard(claim_ids: list[str] | set[str], facts: dict) -> dict:
    """从候选关联的 M1 视觉证据账本提取受支持实体词元、受支持关系端点与不安全词元。"""
    supported_heads: set[str] = set()
    supported_rel_heads: set[str] = set()
    unsafe_heads: set[str] = set()

    for cid in claim_ids or []:
        f = facts.get(cid)
        if not f or f.get("polarity") != "positive":
            continue
        ss = effective_statuses(f)
        cat = f.get("category")
        if cat in {"object", "identity"}:
            concrete, head = is_concrete_entity(f)
            if not concrete:
                continue
            toks = _head_lemmas(head)
            if ss == ["supported", "supported"]:
                supported_heads |= toks
            else:
                unsafe_heads |= toks
        elif cat in {"spatial", "relation"} and ss == ["supported", "supported"]:
            for ent in f.get("entities") or []:
                eh = " ".join(str(ent.get("head") or ent.get("key") or ent.get("anchor") or "").lower().split("[")[0].split())
                if eh and eh not in NON_CONCRETE_HEADS:
                    supported_rel_heads |= _head_lemmas(eh)

    # 若同一词元存在不安全声明，则优先归入 unsafe，避免放行冲突猜测
    supported_heads -= unsafe_heads
    supported_rel_heads -= unsafe_heads
    return {
        "supported_heads": supported_heads,
        "supported_rel_heads": supported_rel_heads,
        "unsafe_heads": unsafe_heads,
    }


def should_protect_span(
    doc,
    text: str,
    current_spans: list[tuple[int, int, str]],
    start: int,
    end: int,
    guard: dict | None,
) -> bool:
    """判定待裁剪片段 [start, end) 是否包含唯一受视觉证据支持的具象物体或关系端点。"""
    if not guard:
        return False
    fragment = text[start:end]
    # 含未消解 or 或显式否定缺失结构的片段不予保护，交由修复规则清理
    if OR_WORD_RE.search(fragment) or NEG_ABSENCE_RE.search(fragment):
        return False
    inside = noun_heads(doc, start, end)
    if inside & guard["unsafe_heads"]:
        return False
    lost = newly_lost_heads(doc, current_spans, start, end)
    protected_vocab = guard["supported_heads"] | guard["supported_rel_heads"]
    if lost & protected_vocab:
        guard["triggered"] = True
        return True
    return False


def candidate_v3_diagnostics(c_feat: dict, facts: dict, gate: dict | None = None, max_r_edge: float = 0.45) -> dict:
    """计算候选描述的具象零支持数、边缘风险预算 R_edge、否定缺失标记与 V3 准入结果。"""
    zero_sup_concrete = 0
    any_ref_obj = 0
    neg_concrete_obj = 0
    entity_best: dict[str, tuple[int, float]] = {}
    ds_objs: set[str] = set()
    ss_objs: set[str] = set()
    ds_rels = 0

    all_pos_objs = []
    for k in c_feat.get("claim_ids", []):
        f = facts.get(k)
        if not f:
            continue
        if f.get("polarity") == "negative" and f.get("category") in {"object", "identity"}:
            conc, _ = is_concrete_entity(f)
            if conc:
                neg_concrete_obj += 1
        elif f.get("polarity") == "positive" and f.get("category") in {"object", "identity"}:
            all_pos_objs.append(f)
        elif f.get("polarity") == "positive" and f.get("category") in {"spatial", "relation"}:
            if effective_statuses(f) == ["supported", "supported"]:
                ds_rels += 1

    if gate:
        for f in (gate.get("pending_checks") or {}).values():
            if f.get("polarity") == "positive" and f.get("category") in {"object", "identity"}:
                all_pos_objs.append(f)

    for f in all_pos_objs:
        ss = effective_statuses(f)
        if "refuted" in ss:
            any_ref_obj += 1
        concrete, head = is_concrete_entity(f)
        if not concrete:
            continue
        nsup = ss.count("supported")
        if nsup == 0:
            zero_sup_concrete += 1
        ptrues = [
            float(ch["p_true"])
            for ch in f.get("checks", [])
            if isinstance(ch.get("p_true"), (int, float)) and not ch.get("low_mass") and ch.get("valid", True)
        ]
        mp = (sum(ptrues) / len(ptrues)) if ptrues else (0.82 if nsup == 2 else 0.58 if nsup == 1 else 0.30)
        prev = entity_best.get(head)
        if prev is None or (nsup, mp) > prev:
            entity_best[head] = (nsup, mp)
        if nsup == 2:
            ds_objs.add(head)
        elif nsup == 1:
            ss_objs.add(head)

    ss_objs -= ds_objs
    # 双模板全支持真物体风险为 0；单模板支持且零反驳（['supported','unclear']）赋予 0.35 边际折算权重，避免误杀多物体高覆盖描述；零支持保留 1.0 全额风险
    r_edge = sum(
        (0.35 if nsup == 1 else 1.0) * max(0.0, 1.0 - mp)
        for nsup, mp in entity_best.values()
        if nsup < 2
    )
    min_concrete_p = min((mp for _, mp in entity_best.values()), default=1.0)
    text = c_feat.get("text", "")
    has_or = bool(OR_WORD_RE.search(text))
    has_neg_clause = bool(NEG_ABSENCE_RE.search(text))

    gate_ok = True if gate is None else bool(gate.get("passed", False))
    core_clean = (
        not c_feat.get("errors")
        and gate_ok
        and any_ref_obj == 0
        and neg_concrete_obj == 0
        and not has_neg_clause
    )
    base_safe = core_clean and zero_sup_concrete == 0 and r_edge <= max_r_edge
    silver_no_or_safe = (
        core_clean
        and (not has_or)
        and zero_sup_concrete <= 1
        and min_concrete_p >= 0.40
        and r_edge <= 0.65
    )
    return {
        "base_safe": base_safe,
        "strict_no_or_safe": base_safe and (not has_or),
        "silver_no_or_safe": silver_no_or_safe,
        "any_ref_obj": any_ref_obj,
        "zero_sup_concrete": zero_sup_concrete,
        "min_concrete_p": round(min_concrete_p, 6),
        "neg_concrete_obj": neg_concrete_obj,
        "has_neg_clause": has_neg_clause,
        "has_or": has_or,
        "r_edge": round(r_edge, 6),
        "n_ds_obj": len(ds_objs),
        "n_ss_obj": len(ss_objs),
        "n_ds_rel": ds_rels,
        "qv": float((c_feat.get("coverage") or {}).get("qv", 0.0) or 0.0),
        "mix": 0.5 * (float((c_feat.get("coverage") or {}).get("f", 0.0) or 0.0) + float((c_feat.get("coverage") or {}).get("s", 0.0) or 0.0)),
    }
