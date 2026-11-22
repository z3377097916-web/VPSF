"""盲标析取结构抽取规则（保留 V2 二元抽取契约，并新增三元及以上 A, B, or C 名词枚举抽取）。"""
import re, statistics


def canonical_name(head):
    tokens = [head]
    for child in head.children:
        if child.dep_ == 'compound':
            tokens.extend(child.subtree)
    return ' '.join(t.text for t in sorted(set(tokens), key=lambda t: t.i))


def check_np(chunk, head):
    det = next((t.text for t in chunk if t.dep_ == 'det' and t.head == head and t.lower_ in {'a', 'an', 'the', 'these', 'those', 'this', 'that'}), '')
    name = canonical_name(head)
    return ((det + ' ') if det else '') + name


def extract(nlp, text, doc=None):
    """二元名词析取抽取（保持与 V2 安全回归测试完全兼容）。"""
    if not re.search(r'\bor\b', text, re.I):
        return []
    doc = nlp(text) if doc is None else doc
    chunks = {c.root.i: c for c in doc.noun_chunks}
    groups = []
    quoted = [m.span() for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”', text)]
    for t in doc:
        if t.dep_ != 'conj' or t.pos_ not in {'NOUN', 'PROPN'} or t.head.pos_ not in {'NOUN', 'PROPN'}:
            continue
        h = t.head
        if h.i >= t.i or h.i not in chunks or t.i not in chunks:
            continue
        if len([h] + list(h.conjuncts)) != 2:
            continue
        middle = list(doc[h.i + 1:t.i])
        cc = [x for x in middle if x.lower_ == 'or' and x.dep_ == 'cc']
        if len(cc) != 1 or any(x.lower_ in {'and', 'but'} or x.text in {',', ';'} for x in middle):
            continue
        a, b = chunks[h.i], chunks[t.i]
        or_token = cc[0]
        while len(a) > 1 and a[0].pos_ in {'ADV', 'PART'}:
            a = doc[a.start + 1:a.end]
        while len(b) > 1 and b[0].pos_ in {'ADV', 'PART'}:
            b = doc[b.start + 1:b.end]
        if any(x.lemma_.lower() in {'half', 'piece', 'part', 'kind', 'type', 'sort', 'variety', 'edge', 'side'} and any(c.dep_ == 'prep' and c.lower_ == 'of' for c in x.children) for x in (h, t)):
            continue
        if a.end_char > or_token.idx or b.start_char < or_token.idx:
            continue
        if re.search(r'\b(no|not|without|neither|nor)\b', t.sent.text, re.I):
            continue
        if any(not (b.end_char <= s or a.start_char >= e) for s, e in quoted):
            continue
        groups.append(dict(
            a=a.text, b=b.text,
            a_span=[a.start_char, a.end_char], b_span=[b.start_char, b.end_char],
            a_head=[h.idx, h.idx + len(h.text)], b_head=[t.idx, t.idx + len(t.text)],
            a_name=canonical_name(h), b_name=canonical_name(t),
            a_check_np=check_np(a, h), b_check_np=check_np(b, t),
            a_plural=h.tag_ in {'NNS', 'NNPS'}, b_plural=t.tag_ in {'NNS', 'NNPS'},
            first=[or_token.idx, b.end_char], last=[a.start_char, b.start_char],
            sentence=t.sent.text,
        ))
    return groups


def extract_multiway_nominal(nlp, text, doc=None):
    """E3-V3 新增：抽取三元及以上名词性 or 并列枚举（如 'A, B, or C'），供多路视觉核查择优单选。"""
    if not re.search(r'\bor\b', text, re.I):
        return []
    doc = nlp(text) if doc is None else doc
    chunks = {c.root.i: c for c in doc.noun_chunks}
    quoted = [m.span() for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”', text)]
    seen_heads = set()
    multi_groups = []

    for h in doc:
        if h.i in seen_heads or h.pos_ not in {'NOUN', 'PROPN'}:
            continue
        conjs = [c for c in h.conjuncts if c.pos_ in {'NOUN', 'PROPN'} and c.i > h.i]
        if len(conjs) < 2:
            continue
        members = sorted([h] + conjs, key=lambda x: x.i)
        seen_heads.update(m.i for m in members)
        if any(m.i not in chunks for m in members):
            continue
        span_tokens = list(doc[members[0].i:members[-1].i + 1])
        has_or = any(x.lower_ == 'or' and x.dep_ == 'cc' for x in span_tokens)
        has_and = any(x.lower_ in {'and', 'but'} for x in span_tokens)
        if not has_or or has_and:
            continue
        if re.search(r'\b(no|not|without|neither|nor)\b', h.sent.text, re.I):
            continue
        item_chunks = []
        valid = True
        for m in members:
            ch = chunks[m.i]
            while len(ch) > 1 and ch[0].pos_ in {'ADV', 'PART'}:
                ch = doc[ch.start + 1:ch.end]
            if len(ch) == 0:
                valid = False
                break
            item_chunks.append((m, ch))
        if not valid:
            continue
        full_start = item_chunks[0][1].start_char
        full_end = item_chunks[-1][1].end_char
        if any(not (full_end <= s or full_start >= e) for s, e in quoted):
            continue
        # 检查相邻项之间仅由逗号/空白/or 连接，避免跨越复杂从句误删
        ok_links = True
        for idx in range(len(item_chunks) - 1):
            gap_str = text[item_chunks[idx][1].end_char:item_chunks[idx + 1][1].start_char]
            if not re.fullmatch(r'\s*(?:,\s*(?:or\s+)?|or\s+)\s*', gap_str, re.I):
                ok_links = False
                break
        if not ok_links:
            continue
        lead_det = next(
            (t.text for t in chunks[members[0].i] if t.dep_ == 'det' and t.head == members[0] and t.lower_ in {'a', 'an', 'the'}),
            '',
        )
        options = []
        for idx, (m, ch) in enumerate(item_chunks):
            opt_text = ch.text
            # 若首项带冠词而后续并列项省略冠词（如 'a chair, bench, or couch'），在替换时自动继承合理表面形式
            has_own_det = any(t.dep_ == 'det' for t in ch)
            surface = opt_text
            if idx > 0 and lead_det and not has_own_det and m.tag_ not in {'NNS', 'NNPS'}:
                if lead_det.lower() in {'a', 'an'}:
                    article = 'an' if re.match(r'[aeiou]', opt_text, re.I) else 'a'
                    if lead_det[0].isupper():
                        article = article.capitalize()
                    surface = f"{article} {opt_text}"
                else:
                    surface = f"{lead_det} {opt_text}"
            options.append(dict(
                text=opt_text,
                replacement=surface,
                name=canonical_name(m),
                check_np=check_np(ch, m),
                span=[ch.start_char, ch.end_char],
            ))
        multi_groups.append(dict(
            kind='multiway_nominal_or',
            full_span=[full_start, full_end],
            options=options,
            sentence=h.sent.text,
        ))
    return multi_groups
