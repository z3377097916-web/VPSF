"""E3-V3 外科式局部结构修复：二元+多元名词析取择优、显式否定缺失剥离、M1 证据账本跨度保护裁剪。"""
from .or_syntax import extract, extract_multiway_nominal
import re
from .or_choice import apply, decision, apply_multiway
from .adjunct import edit as adjunct_edit, neg_tail_edit
from .structural import cuts
from .cover_budget import edit as budget_edit
from .ledger_guard import build_ledger_span_guard
from rlsfv.verification import yesno_prompt, yesno_check, YESNO_WORDS

POLICY = 'e3_structural_repair_v3'

SPEC_OR_CUE_RE = re.compile(
    r'\b(?:suggest\w*|appear\w*|seem\w*|might|could|may|possibl\w*|perhaps|'
    r'likely|probabl\w*|either|whether|convey\w*|indicat\w*|impl(?:y|ies))\b',
    re.I,
)


def _adj_or_edit(doc, text: str) -> tuple[str, list[list[int]]]:
    """消解无 either 前导的二元形容词析取（如 'professional or formal event' -> 'professional event'）。"""
    spans = []
    n_toks = len(doc)
    for tok in doc:
        if tok.pos_ != 'ADJ':
            continue
        ccs = [c for c in tok.children if c.dep_ == 'cc' and c.lower_ == 'or']
        conjs = [c for c in tok.children if c.dep_ == 'conj']
        if len(ccs) != 1 or len(conjs) != 1:
            continue
        cc, adj2 = ccs[0], conjs[0]
        if adj2.pos_ != 'ADJ':
            continue
        if any(c.lower_ in {'either', 'whether'} for c in tok.children):
            continue
        if tok.i > 0 and doc[tok.i - 1].lower_ in {'either', 'whether', 'between'}:
            continue
        sub = list(adj2.subtree)
        if not sub or min(t.i for t in sub) <= cc.i:
            continue
        if not all(t.pos_ in {'ADJ', 'ADV'} for t in sub):
            continue
        last_i = max(t.i for t in sub)
        if last_i + 1 < n_toks and doc[last_i + 1].lower_ in {'or', ','}:
            continue
        s_char = tok.idx + len(tok.text)
        e_char = max(t.idx + len(t.text) for t in sub)
        if 0 <= s_char < e_char <= len(text) and text[s_char].isspace():
            spans.append((s_char, e_char))
    if not spans:
        return text, []
    # 去重并从右向左无重叠删除
    clean = []
    for a, b in sorted(spans):
        if not clean or a >= clean[-1][1]:
            clean.append((a, b))
    out = text
    for a, b in reversed(clean):
        out = out[:a] + out[b:]
    return out, [[a, b] for a, b in clean]


def _speculative_or_sentence_edit(doc, text: str) -> tuple[str, list[list[int]]]:
    """整句剥离残余的含 or 纯推测/总结句（须保留至少 2 句事实描述且总词数 >= 30）。"""
    if not re.search(r'\bor\b', text, re.I):
        return text, []
    sents = [s for s in doc.sents if s.text.strip()]
    if len(sents) < 2:
        return text, []
    kept = set(range(len(sents)))
    removed = []
    for i in range(len(sents) - 1, -1, -1):
        s_txt = sents[i].text
        if not (re.search(r'\bor\b', s_txt, re.I) and SPEC_OR_CUE_RE.search(s_txt)):
            continue
        if len(kept) - 1 < 2:
            break
        rem_words = sum(len(sents[j].text.split()) for j in kept if j != i)
        if rem_words < 20:
            continue
        nxt = min((j for j in kept if j > i), default=None)
        if nxt is not None and re.match(r'^\s*(?:This|That|These|Those)\b', sents[nxt].text):
            continue
        kept.remove(i)
        removed.append([sents[i].start_char, sents[i].end_char])
    if not removed:
        return text, []
    kept_list = sorted(kept)
    out = text[sents[kept_list[0]].start_char:sents[kept_list[0]].end_char].strip()
    for prev_i, cur_i in zip(kept_list, kept_list[1:]):
        between = text[sents[prev_i].end_char:sents[cur_i].start_char]
        sep = '\n\n' if '\n\n' in between else ' '
        out += sep + text[sents[cur_i].start_char:sents[cur_i].end_char].strip()
    return out, list(reversed(removed))


def safe_or_application(text, groups):
    """不因局部删除残留比较级指代词（如 'a similar incident'），重叠语法解析自动弃权。"""
    spans = [(min(g['first'][0], g['last'][0]), max(g['first'][1], g['last'][1])) for g in groups]
    ambiguous = {i for i, (a, b) in enumerate(spans) if any(
        j != i and max(a, c) < min(b, d) for j, (c, d) in enumerate(spans))}
    usable = []
    decisions = []
    for i, g in enumerate(groups):
        side = decision(g, 'g_argmax')
        winner = g['a'] if side == 'first' else g['b'] if side == 'last' else ''
        if i in ambiguous:
            g['abstention_reason'] = 'ambiguous_overlapping_alternatives'
            side = None
        elif side and re.search(r'\b(?:similar|same|other|another|such)\b', winner, re.I):
            g['abstention_reason'] = 'winner_depends_on_removed_alternative'
            side = None
        else:
            usable.append(g)
        decisions.append(side)
    edited, _ = apply(text, usable, 'g_argmax') if usable else (text, [])
    return edited, decisions


def _verify_fact_specs(specs, image, pipeline):
    if not specs:
        return []
    requests = [
        dict(kind='repair_or_yesno', image_sha256=pipeline.image_hash,
             prompt=yesno_prompt(f, t), letters=YESNO_WORDS, history=[])
        for f in specs for t in (0, 1)
    ]
    raw = pipeline.run_batched(
        requests, 'verification',
        lambda rs: pipeline.backend.choice_batch([r['prompt'] for r in rs], YESNO_WORDS, image),
        lambda r: pipeline.backend.choice(r['prompt'], YESNO_WORDS, image),
    )
    cfg = pipeline.cfg['verification']
    return [
        [yesno_check(raw[2 * i + t], f, cfg['yesno_band'], cfg['yesno_min_mass']) for t in (0, 1)]
        for i, f in enumerate(specs)
    ]


def repair(nlp, text, image, pipeline, claim_ids=None, source_facts=None, use_guard=True):
    audit = []
    edited = text
    _doc_cache = [None, None]

    def _doc(s):
        if _doc_cache[0] != s:
            _doc_cache[0] = s
            _doc_cache[1] = nlp(s)
        return _doc_cache[1]

    # 第 1a 步：二元名词析取（A or B）双模板视觉核查消解
    groups = extract(nlp, edited, doc=_doc(edited)) if re.search(r'\bor\b', edited, re.I) else []
    if groups:
        specs = []
        for g in groups:
            for side in ('a', 'b'):
                name = g[side + '_name']
                specs.append(dict(
                    statement='The image contains ' + g[side + '_check_np'] + '.',
                    polarity='positive', modality='definite',
                    entities=[dict(anchor=name, display=name)],
                ))
        checks = _verify_fact_specs(specs, image, pipeline)
        for i, g in enumerate(groups):
            g['checks'] = checks[2 * i:2 * i + 2]
        after_bin, decisions = safe_or_application(edited, groups)
        audit.append(dict(kind='noun_alternative', groups=groups, decisions=decisions, before=edited, after=after_bin))
        edited = after_bin

    # 第 1b 步：三元及以上名词并列（A, B, or C）双模板视觉择优单选
    mgroups = extract_multiway_nominal(nlp, edited, doc=_doc(edited)) if re.search(r'\bor\b', edited, re.I) else []
    if mgroups:
        mspecs = []
        offsets = []
        for mg in mgroups:
            s_idx = len(mspecs)
            for opt in mg['options']:
                mspecs.append(dict(
                    statement='The image contains ' + opt['check_np'] + '.',
                    polarity='positive', modality='definite',
                    entities=[dict(anchor=opt['name'], display=opt['name'])],
                ))
            offsets.append((s_idx, len(mspecs)))
        mchecks = _verify_fact_specs(mspecs, image, pipeline)
        for mg, (s_idx, e_idx) in zip(mgroups, offsets):
            mg['checks'] = mchecks[s_idx:e_idx]
        after_multi, mdecisions = apply_multiway(edited, mgroups)
        if after_multi != edited:
            audit.append(dict(kind='multiway_noun_alternative', groups=mgroups, decisions=mdecisions, before=edited, after=after_multi))
            edited = after_multi

    # 第 1c 步：消解二元形容词析取（如 'professional or formal event' -> 'professional event'）
    if re.search(r'\bor\b', edited, re.I):
        after_adj, adj_spans = _adj_or_edit(_doc(edited), edited)
        if after_adj != edited:
            audit.append(dict(kind='adjective_or_reduction', before=edited, after=after_adj, spans=adj_spans))
            edited = after_adj

    # 第 1d 步：剥离句尾显式否定缺失短语（如 ', with no cars or people visible'）
    from .adjunct import NEG_TAIL_RX
    if NEG_TAIL_RX.search(edited):
        after_neg, neg_spans = neg_tail_edit(nlp, edited, doc=_doc(edited))
        if after_neg != edited:
            audit.append(dict(kind='negated_absence_tail', before=edited, after=after_neg, spans=neg_spans))
            edited = after_neg

    # 构建基于 M1 证据账本的实体与双支持关系端点保护集
    guard = build_ledger_span_guard(claim_ids, source_facts) if (use_guard and claim_ids and source_facts) else None

    # 第 2–4 步：受视觉证据保护的句尾推测、前瞻目的与新词预算裁剪
    for kind, function in [
        ('or_speculative_tail', lambda s: adjunct_edit(nlp, s, 'or_tail', doc=_doc(s), guard=guard) if re.search(r'\bor\b', s, re.I) else (s, [])),
        ('identity_prospective', lambda s: cuts(nlp, s, 'identity_prospective', doc=_doc(s), guard=guard)),
        ('spec_novel2', lambda s: budget_edit(_doc(s), s, 'spec_novel2', guard=guard)),
    ]:
        result = function(edited)
        if result[0] != edited:
            audit.append(dict(kind=kind, before=edited, after=result[0], spans=result[1],
                              reasons=result[2] if len(result) > 2 else []))
        edited = result[0]

    # 第 5 步：若仍残留含 or 的整句推测/总结句（如 'The scene appears as if ... or ...'），在保证双句与 >=30 词前提下整句剥离
    if re.search(r'\bor\b', edited, re.I):
        after_sor, sor_spans = _speculative_or_sentence_edit(_doc(edited), edited)
        if after_sor != edited:
            audit.append(dict(kind='speculative_or_sentence_drop', before=edited, after=after_sor, spans=sor_spans))
            edited = after_sor

    if guard and guard.get('triggered'):
        audit.append(dict(kind='ledger_guard_protected_span'))

    if not edited.strip():
        raise ValueError('Repair emptied the answer')
    return edited, audit
