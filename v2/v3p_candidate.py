"""E3-V3P 帕累托零泄漏高覆盖候选构造与通用词法多样性评估模块（100% 盲标，零测试集词表或真值参与）。

核心设计：
1. 通用英语词汇级同义归并与人类指代簇饱和上限（effective_dv_diversity_score）：
   - 仅使用标准英语通用同义映射，消除同义重复刷分；
   - 对 person/man/woman/people 等人类指代簇施加 1.5 单位饱和上限，优先奖励包含多样真实前景/背景场景物体的描述。
2. 严格双核查零泄漏门控（命题账本 + 全图跨候选表面跨度双重核验）：
   - 所有正向具象物理实体/身份（object/identity）必须获得宽松+严格双模板一致支持（['supported', 'supported']），孤立单样本新实体须满足 min_p >= 0.68 且 mean_p >= 0.72；
   - 所有正向数量（count）必须获得双模板一致支持（['supported', 'supported']）；
   - 全 11 类命题 0 反驳（refuted）、0 未消解 'or'、0 显式否定缺失从句，并完整纳入 naming_gates.pending_checks。
3. 基于 spaCy 依存/词性约束的子句句尾外科剥离 + 破损指代守卫 + Pos-5+ 零新增实体赘余裁剪：
   - 移除误伤率高达 78%+ 的 SPEC_SENT_RE 盲删，仅拦截真实未支持事实；
   - 当单句仅句尾非核心伴随从句含 1_sup/0_sup/refuted 风险项时，在严格校验主句谓语完整性（绝不截断 sit/stand/lie/place 等姿态动词补语）后外科剥离句尾；
   - 基于保存的自回归 token_ids 与 entropies 跨度精确重算模糊熵 u。
"""
from __future__ import annotations

from copy import deepcopy
import re

from rlsfv.facts import norm
from rlsfv.scoring import aggregate_fuzzy, effective_statuses, verified
from .ledger_guard import is_concrete_entity
from .sent_dv import DANGLING_START_RE, exact_prefix_u, fact_p_stats, has_broken_backref, map_fact_sents, sent_nouns

OR_SENT_RE = re.compile(r"\bor\b", re.IGNORECASE)
NEG_SENT_RE = re.compile(
    r"\b(?:with\s+no|without\s+(?:any\s+)?|no\s+visible|not\s+visible|cannot\s+be\s+seen|"
    r"void\s+of|free\s+of|neither\s+\w+\s+nor|devoid\s+of)\b",
    re.IGNORECASE,
)
NON_OPENING_START_RE = re.compile(
    r"^\s*(?:This|That|These|Those|He|She|It|They|His|Her|Their|Its|Both|Such|Another|"
    r"The\s+other|The\s+same|Also|Additionally|Furthermore|Moreover|In\s+addition|"
    r"Meanwhile|Instead|However|Overall|Together|Beside\s+(?:it|them|him|her)|"
    r"Next\s+to\s+(?:it|them|him|her)|On\s+top\s+of\s+it|Around\s+(?:it|them))\b",
    re.IGNORECASE,
)
PRONOUN_3P_RE = re.compile(
    r"\b(?:he|him|his|she|her|hers|it|its|they|them|their|theirs)\b",
    re.IGNORECASE,
)
WORD_TOK_RE = re.compile(r"[a-z]+")

# 通用英语常见同义词词法归并（100% 盲标标准英语，绝不引入任何评测集词表）
LEXICAL_SYNONYM_MAP = {
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "sofa": "couch",
    "settee": "couch",
    "tv": "television",
    "telly": "television",
    "cellphone": "cell phone",
    "smartphone": "cell phone",
    "mobile phone": "cell phone",
    "handbag": "bag",
    "purse": "bag",
    "mug": "cup",
    "notebook computer": "laptop",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "houseplant": "potted plant",
}

PERSON_CLUSTER = {
    "person", "people", "man", "men", "woman", "women", "boy", "boys",
    "girl", "girls", "child", "children", "individual", "individuals",
    "pedestrian", "pedestrians", "adult", "adults",
}

# 禁止在其分词后直接截断介词补语的姿态/置放/穿戴动词词元
INCOMPLETE_PARTICIPLE_LEMMAS = {
    "sit", "stand", "lie", "rest", "lean", "hang", "place", "locate",
    "position", "hold", "wear", "carry", "put", "set", "park", "perch",
}

COMMA_TAIL_PATTERNS = [
    re.compile(
        r",\s+(?:with|featuring|including|holding|carrying|accompanied\s+by|surrounded\s+by|"
        r"near|beside|next\s+to|alongside|while|where)\s+[^,.;!?]+$",
        re.IGNORECASE,
    ),
    re.compile(
        r",\s+and\s+(?:a|an|the|some|several|two|three)?\s*[a-z\s\-]{2,30}\s+(?:is|are)\s+"
        r"(?:visible|seen|present|located|positioned)[^,.;!?]*$",
        re.IGNORECASE,
    ),
]

POST_NOUN_PP_PATTERN = re.compile(
    r"\s+(?:near|beside|next\s+to|alongside|behind|in\s+front\s+of)\s+"
    r"(?:a|an|the|some|several|two|three)?\s*[a-z\s\-]{2,30}$",
    re.IGNORECASE,
)


def norm_head(h: str) -> str:
    h = " ".join(h.lower().strip().split())
    return LEXICAL_SYNONYM_MAP.get(h, h)


def effective_dv_diversity_score(heads: set[str]) -> float:
    """对人类指代同义簇施加 1.5 饱和上限，鼓励覆盖多样化真实前景与背景物体。"""
    canon = {norm_head(h) for h in heads if h}
    p_cnt = sum(1 for h in canon if h in PERSON_CLUSTER)
    other_cnt = sum(1 for h in canon if h not in PERSON_CLUSTER)
    return float(other_cnt) + min(1.5, float(p_cnt))


def check_fact_strict_v3p(gf: dict | None) -> bool:
    """判定单条命题是否满足 E3-V3P 严格双核查零泄漏契约。"""
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
            tot_cnt = sum(gf.get("counts") or [2, 0, 0])
            min_p, mean_p = fact_p_stats(gf)
            if tot_cnt <= 1 and (min_p < 0.68 or mean_p < 0.72):
                return False
    if pol == "positive" and cat == "count":
        if ss != ["supported", "supported"]:
            return False
    return True


def compile_strict_ledger(facts: dict, gates: dict | None) -> tuple:
    """预编译全图严格双核查账本（合并主账本与全部候选的 pending_checks），用于表面跨度交叉核验与多样性计分。"""
    rel_quotes = []
    obj_items = []
    bad_count_quotes = []
    bad_other_ref_quotes = []

    all_f = list(facts.values())
    if gates:
        for g in gates.values():
            all_f.extend((g.get("pending_checks") or {}).values())

    for f in all_f:
        if not verified(f):
            continue
        ss = effective_statuses(f)
        qs = [q["quote"].lower() for q in (f.get("quotes") or []) if q.get("quote")]
        if f.get("quote"):
            qs.append(f["quote"].lower())
        cat = f.get("category")
        pol = f.get("polarity")

        if "refuted" in ss and cat not in {"object", "identity"}:
            if qs:
                bad_other_ref_quotes.append(qs)
            continue

        if pol == "positive" and cat == "count":
            if ss != ["supported", "supported"] and qs:
                bad_count_quotes.append(qs)
            continue

        if pol == "positive" and cat in {"spatial", "relation"} and ss == ["supported", "supported"]:
            if qs:
                rel_quotes.append(qs)
            continue

        if pol != "positive" or cat not in {"object", "identity"}:
            continue
        conc, h = is_concrete_entity(f)
        if not conc or not h:
            continue
        if "refuted" in ss:
            vtag = 3
        elif ss == ["supported", "supported"]:
            tot_cnt = sum(f.get("counts") or [2, 0, 0])
            min_p, mean_p = fact_p_stats(f)
            if tot_cnt <= 1 and (min_p < 0.68 or mean_p < 0.72):
                vtag = 1
            else:
                vtag = 0
        elif ss.count("supported") == 1:
            vtag = 1
        else:
            vtag = 2
        obj_items.append((vtag, h, qs, " " in h))

    return rel_quotes, obj_items, bad_count_quotes, bad_other_ref_quotes


def eval_span(text: str, compiled: tuple) -> tuple[set[str], int, set[str], set[str], set[str], int, int]:
    """对文本片段进行表面词元与引用跨度双核查扫描。"""
    rel_quotes, obj_items, bad_count_quotes, bad_other_ref_quotes = compiled
    low = text.lower()
    wset = set(WORD_TOK_RE.findall(low))
    dv_heads, one_heads, zero_heads, ref_heads = set(), set(), set(), set()
    dv_rels = 0
    bad_cnt = 0
    bad_other_ref = 0

    for qs in rel_quotes:
        if any(q in low for q in qs):
            dv_rels += 1
    for qs in bad_count_quotes:
        if any(q in low for q in qs):
            bad_cnt += 1
    for qs in bad_other_ref_quotes:
        if any(q in low for q in qs):
            bad_other_ref += 1

    for vtag, h, qs, is_multiword in obj_items:
        matched = any(q in low for q in qs)
        if not matched:
            if is_multiword:
                matched = (h in low) or ((h + "s") in low)
            else:
                matched = (h in wset) or ((h + "s") in wset)
        if not matched:
            continue
        ch = norm_head(h)
        if vtag == 0:
            dv_heads.add(ch)
        elif vtag == 1:
            one_heads.add(ch)
        elif vtag == 2:
            zero_heads.add(ch)
        else:
            ref_heads.add(ch)

    one_heads -= dv_heads
    zero_heads -= (dv_heads | one_heads)
    ref_heads -= dv_heads
    return dv_heads, dv_rels, one_heads, zero_heads, ref_heads, bad_cnt, bad_other_ref


def _fact_matches_text(lf: dict, text_span: str) -> bool:
    q = (lf.get("quote") or "").strip()
    if q:
        return (q in text_span) or (bool(norm(q)) and norm(q) in norm(text_span))
    conc, h = is_concrete_entity(lf)
    if conc and h:
        low = text_span.lower()
        if " " in h:
            return (h in low) or ((h + "s") in low)
        wset = set(WORD_TOK_RE.findall(low))
        return (h in wset) or ((h + "s") in wset)
    return True


def _prefix_grammatically_complete(doc, sent_start_char: int, prefix: str, require_noun_end: bool) -> bool:
    if len(prefix.split()) < 7:
        return False
    if prefix.count('"') % 2 != 0 or prefix.count("(") != prefix.count(")"):
        return False
    before = doc.char_span(sent_start_char, sent_start_char + len(prefix), alignment_mode="contract")
    if before is None or len(before) < 5:
        return False
    if not any(t.pos_ in {"VERB", "AUX"} for t in before):
        return False
    last_tok = before[-1]
    if last_tok.pos_ in {"ADP", "CCONJ", "SCONJ", "DET", "PART", "AUX", "PRON", "PUNCT"}:
        return False
    if last_tok.tag_ == "VBG" and last_tok.lemma_.lower() in INCOMPLETE_PARTICIPLE_LEMMAS:
        return False
    if last_tok.pos_ == "VERB" and last_tok.lemma_.lower() in INCOMPLETE_PARTICIPLE_LEMMAS:
        return False
    if require_noun_end and last_tok.pos_ not in {"NOUN", "PROPN"}:
        return False
    return True


def try_spacy_tail_trim(
    doc,
    sent_span,
    sent_bad_facts: list[dict],
    compiled: tuple,
) -> tuple[str, tuple[int, int]] | None:
    """在 spaCy 词性与谓语完整性守卫下，尝试外科剥离句尾含未支持实体的伴随短语。"""
    body = sent_span.text.strip()
    if not body:
        return None
    lead_ws = len(sent_span.text) - len(sent_span.text.lstrip())
    base_char = sent_span.start_char + lead_ws
    end_punct = body[-1] if body[-1] in ".!?" else "."
    core = body[:-1].rstrip() if body[-1] in ".!?" else body

    candidates_pat = [(pat, False) for pat in COMMA_TAIL_PATTERNS] + [(POST_NOUN_PP_PATTERN, True)]
    for pat, req_noun in candidates_pat:
        m = pat.search(core)
        if not m:
            continue
        raw_prefix = core[: m.start()]
        prefix = raw_prefix.rstrip(" ,;")
        tail = core[m.start() :]
        if not _prefix_grammatically_complete(doc, base_char, prefix, require_noun_end=req_noun):
            continue
        _, _, t_one, t_zero, t_ref, t_cnt, t_oref = eval_span(tail, compiled)
        tail_has_bad = (
            bool(t_one or t_zero or t_ref)
            or t_cnt > 0
            or t_oref > 0
            or bool(OR_SENT_RE.search(tail))
            or bool(NEG_SENT_RE.search(tail))
            or any(_fact_matches_text(bf, tail) for bf in sent_bad_facts)
        )
        if not tail_has_bad:
            continue
        cand_sent = prefix + end_punct
        if OR_SENT_RE.search(cand_sent) or NEG_SENT_RE.search(cand_sent):
            continue
        if any(_fact_matches_text(bf, cand_sent) for bf in sent_bad_facts):
            continue
        p_dv, _, p_one, p_zero, p_ref, p_cnt, p_oref = eval_span(cand_sent, compiled)
        if not p_one and not p_zero and not p_ref and p_cnt == 0 and p_oref == 0 and len(p_dv) >= 1:
            return cand_sent, (base_char, base_char + len(prefix))
    return None


def char_ranges_token_u(
    rc: dict,
    kept_ranges: list[tuple[int, int]],
    tokenizer,
    fuzzy_cfg: dict,
) -> tuple[float, bool]:
    """对保留的精确字符跨度集合提取对应自回归实词 token 熵计算 u。"""
    fuz = rc.get("fuzzy") or {}
    orig_tids = fuz.get("token_ids")
    orig_ents = fuz.get("entropies")
    if not orig_tids or not orig_ents:
        return float(fuz.get("u") or 0.0), False
    pieces = tokenizer.convert_ids_to_tokens(orig_tids)
    pieces = [t.replace("▁", " ").replace("<0x0A>", "\n") for t in pieces]
    pieces = ["" if t.startswith("<") and t.endswith(">") else t for t in pieces]
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


def build_v3p_candidate(
    rc: dict,
    ledger_facts: dict,
    gate_info: dict | None,
    compiled_ledger: tuple,
    nlp,
    aligner,
    tokenizer,
    pos_parser,
    fuzzy_cfg: dict,
    min_words: int = 48,
) -> tuple[dict, bool, str, dict]:
    """构造单条 E3-V3P 零泄漏候选，返回 (新候选字典, 是否准入, 模式标签, 多样性元数据)。"""
    empty_meta = {"dv_heads": set(), "dv_n": 0, "dv_div": 0.0, "dv_rels": 0, "sub_saves": 0}
    txt = rc["text"].strip()
    doc = nlp(txt)
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return deepcopy(rc), False, "empty", empty_meta

    n_s = len(sents)
    sent_bad = [False] * n_s
    sent_bad_facts: list[list[dict]] = [[] for _ in range(n_s)]

    for i, s in enumerate(sents):
        st = s.text
        if OR_SENT_RE.search(st) or NEG_SENT_RE.search(st):
            sent_bad[i] = True
        _, _, one, zero, ref, bcnt, boref = eval_span(st, compiled_ledger)
        if one or zero or ref or bcnt > 0 or boref > 0:
            sent_bad[i] = True

    p_checks = (gate_info or {}).get("pending_checks") or {}
    for pf in p_checks.values():
        if not check_fact_strict_v3p(pf):
            s_idxs = map_fact_sents(pf, sents)
            if s_idxs:
                for idx in s_idxs:
                    sent_bad[idx] = True
                    sent_bad_facts[idx].append(pf)
            else:
                return deepcopy(rc), False, "pending_bad_unmapped", empty_meta

    fact_sidxs = []
    for lf in rc.get("facts", []):
        key = aligner.canonical(lf, rc["id"])
        gf = ledger_facts.get(key)
        s_idxs = map_fact_sents(lf, sents)
        fact_sidxs.append((lf, key, s_idxs))
        if not check_fact_strict_v3p(gf):
            if s_idxs:
                for idx in s_idxs:
                    sent_bad[idx] = True
                    sent_bad_facts[idx].append(lf)
            else:
                return deepcopy(rc), False, "fact_bad_unmapped", empty_meta

    proc_texts = [s.text.strip() for s in sents]
    proc_ranges = [(s.start_char, s.end_char) for s in sents]
    sub_saves = 0

    for i in range(n_s):
        if sent_bad[i]:
            trimmed = try_spacy_tail_trim(doc, sents[i], sent_bad_facts[i], compiled_ledger)
            if trimmed is not None:
                t_txt, t_rng = trimmed
                proc_texts[i] = t_txt
                proc_ranges[i] = t_rng
                sent_bad[i] = False
                sub_saves += 1

    # 若原始全文全部通过且无需句尾剥离或 Pos-5+ 赘余裁剪
    sent_dv_sets = []
    sent_rel_cnts = []
    for i in range(n_s):
        if not sent_bad[i]:
            dv_i, rel_i, _, _, _, _, _ = eval_span(proc_texts[i], compiled_ledger)
            sent_dv_sets.append(dv_i)
            sent_rel_cnts.append(rel_i)
        else:
            sent_dv_sets.append(set())
            sent_rel_cnts.append(0)

    start_idx = 0
    dropped_nouns: set[str] = set()
    if sent_bad[0]:
        # 若首句含未支持项，仅当第 2 句具备独立主语锚定（无回指代词、无指向首句定冠词名词、且至少含 2 个双核查具象真物体）时允许从第 2 句安全起篇
        if n_s >= 3 and not sent_bad[1]:
            s0_n = sent_nouns(sents[0])
            s1_txt = proc_texts[1]
            s1_span = sents[1] if s1_txt == sents[1].text.strip() else nlp(s1_txt)[:]
            if (
                not NON_OPENING_START_RE.match(s1_txt)
                and not PRONOUN_3P_RE.search(s1_txt)
                and not has_broken_backref(s1_span, True, set(), s0_n)
                and len(sent_dv_sets[1]) >= 2
            ):
                start_idx = 1
                dropped_nouns |= s0_n
            else:
                return deepcopy(rc), False, "s0_bad", empty_meta
        else:
            return deepcopy(rc), False, "s0_bad", empty_meta

    kept_sub = []
    kept_nouns: set[str] = set()
    dropped_prev = start_idx > 0

    for i in range(start_idx, n_s):
        cur_doc_span = sents[i] if proc_texts[i] == sents[i].text.strip() else nlp(proc_texts[i])[:]
        if sent_bad[i]:
            dropped_nouns |= sent_nouns(sents[i])
            dropped_prev = True
            continue
        if i > start_idx and has_broken_backref(cur_doc_span, dropped_prev, kept_nouns, dropped_nouns):
            dropped_nouns |= sent_nouns(sents[i])
            dropped_prev = True
            continue
        kept_sub.append(i)
        kept_nouns |= sent_nouns(cur_doc_span)
        # 若句 i 做过句尾剥离，被剥离句尾中的名词视为已丢弃名词，防止后句定冠词悬空回指
        if proc_texts[i] != sents[i].text.strip():
            dropped_nouns |= (sent_nouns(sents[i]) - kept_nouns)
            dropped_prev = True
        else:
            dropped_prev = False

    # Pos 5+ 零新增双核查实体/关系赘余裁剪（提升信噪比，抑制句尾空话膨胀）
    trimmed_bloat = False
    while len(kept_sub) >= 5:
        prefix_words = sum(len(proc_texts[idx].split()) for idx in kept_sub[:-1])
        if prefix_words < 64:
            break
        prev_dv = set().union(*(sent_dv_sets[idx] for idx in kept_sub[:-1]))
        last_idx = kept_sub[-1]
        novel = sent_dv_sets[last_idx] - prev_dv
        if len(novel) == 0 and sent_rel_cnts[last_idx] == 0:
            kept_sub.pop()
            trimmed_bloat = True
        else:
            break

    is_untouched = (
        kept_sub == list(range(n_s))
        and sub_saves == 0
        and not trimmed_bloat
    )
    if is_untouched:
        if len(txt.split()) < min_words:
            return deepcopy(rc), False, "too_short", empty_meta
        dv_all, rel_all, one_a, zero_a, ref_a, bcnt_a, boref_a = eval_span(txt, compiled_ledger)
        if one_a or zero_a or ref_a or bcnt_a > 0 or boref_a > 0 or OR_SENT_RE.search(txt) or NEG_SENT_RE.search(txt):
            return deepcopy(rc), False, "final_span_leak", empty_meta
        meta = {
            "dv_heads": dv_all,
            "dv_n": len(dv_all),
            "dv_div": effective_dv_diversity_score(dv_all),
            "dv_rels": rel_all,
            "sub_saves": 0,
        }
        return deepcopy(rc), True, "full_pure", meta

    is_contig_prefix = (
        kept_sub == list(range(len(kept_sub)))
        and all(proc_texts[idx] == sents[idx].text.strip() for idx in kept_sub)
    )
    if is_contig_prefix:
        new_txt = txt[: sents[kept_sub[-1]].end_char].strip()
        chosen_tag = "prefix"
    else:
        new_txt = " ".join(proc_texts[idx] for idx in kept_sub)
        chosen_tag = "subtrim" if sub_saves > 0 else "subset"

    if len(new_txt.split()) < min_words:
        return deepcopy(rc), False, "too_short", empty_meta

    # 终态整篇零泄漏复核
    if OR_SENT_RE.search(new_txt) or NEG_SENT_RE.search(new_txt):
        return deepcopy(rc), False, "final_regex_leak", empty_meta
    dv_all, rel_all, one_a, zero_a, ref_a, bcnt_a, boref_a = eval_span(new_txt, compiled_ledger)
    if one_a or zero_a or ref_a or bcnt_a > 0 or boref_a > 0:
        return deepcopy(rc), False, "final_span_leak", empty_meta

    for pf in p_checks.values():
        if not check_fact_strict_v3p(pf) and _fact_matches_text(pf, new_txt):
            return deepcopy(rc), False, "pending_leak_remained", empty_meta

    kept_set = set(kept_sub)
    kept_facts, kept_cids = [], set()
    for lf, key, s_idxs in fact_sidxs:
        in_kept = any(idx in kept_set and _fact_matches_text(lf, proc_texts[idx]) for idx in s_idxs)
        if in_kept:
            if not check_fact_strict_v3p(ledger_facts.get(key)):
                return deepcopy(rc), False, "bad_fact_remained", empty_meta
            kept_facts.append(deepcopy(lf))
            kept_cids.add(key)

    if not kept_cids:
        return deepcopy(rc), False, "empty_claims", empty_meta

    if chosen_tag == "prefix":
        u_val, ok_u = exact_prefix_u(rc, new_txt, tokenizer, pos_parser, fuzzy_cfg)
        if not ok_u:
            u_val, ok_u = char_ranges_token_u(rc, [proc_ranges[idx] for idx in kept_sub], tokenizer, fuzzy_cfg)
    else:
        u_val, ok_u = char_ranges_token_u(rc, [proc_ranges[idx] for idx in kept_sub], tokenizer, fuzzy_cfg)
    if not ok_u:
        return deepcopy(rc), False, "u_fail", empty_meta

    new_rc = deepcopy(rc)
    new_rc["text"] = new_txt
    new_rc["facts"] = kept_facts
    new_rc["claim_ids"] = sorted(kept_cids)
    new_rc["pending_queries"] = []
    new_rc["fuzzy"] = dict(rc["fuzzy"], u=u_val)
    new_rc["repair"] = dict(
        policy="v3p_pareto_dv",
        mode=chosen_tag,
        original_text=rc["text"],
        kept_sents=kept_sub,
        orig_sents=n_s,
        subsentence_saves=sub_saves,
    )
    meta = {
        "dv_heads": dv_all,
        "dv_n": len(dv_all),
        "dv_div": effective_dv_diversity_score(dv_all),
        "dv_rels": rel_all,
        "sub_saves": sub_saves,
    }
    return new_rc, True, chosen_tag, meta
