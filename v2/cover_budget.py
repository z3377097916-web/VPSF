"""盲标覆盖预算控制裁剪（E3-V3 集成 M1 证据账本保护：禁止误裁双模板/单模板已核实的真物体与关系）。"""
import re, statistics

INFER = r'suggesting|indicating|implying|suggests|indicates|implies'
STRONG = r'possibly|probably|perhaps'
ALL = STRONG + '|likely|' + INFER
TAIL = re.compile(r',\s+(?:(?:which|that)\s+)?(?P<cue>' + ALL + r')\b', re.I)
UNCERTAIN = re.compile(r'\b(?:' + ALL + r'|might|may|could)\b', re.I)
BACKGROUND = re.compile(r'\b(?:background|distance|distant|behind|in addition|additionally|furthermore)\b', re.I)


def apply_spans(text, spans):
    merged = []
    for a, b, kind in sorted(spans):
        if merged and a < merged[-1][1]:
            old = merged.pop()
            merged.append((old[0], max(b, old[1]), old[2] + '+' + kind))
        else:
            merged.append((a, b, kind))
    for a, b, kind in reversed(merged):
        left = text[:a].rstrip()
        right = text[b:]
        if kind.startswith('tail'):
            left = left.rstrip(' ,;') + '.'
        gap = ' ' if left and right and not right[0].isspace() else ''
        text = left + gap + right
    return text.strip(), merged


def noun_heads(doc, a, b):
    return {t.lemma_.lower() for t in doc if a <= t.idx < b and t.pos_ in {'NOUN', 'PROPN'} and t.dep_ != 'compound'}


def newly_lost_heads(doc, spans, a, b):
    before = {t.lemma_.lower() for t in doc if t.pos_ in {'NOUN', 'PROPN'} and t.dep_ != 'compound'
              and not any(x <= t.idx < y for x, y, _ in spans)}
    after = {t.lemma_.lower() for t in doc if t.pos_ in {'NOUN', 'PROPN'} and t.dep_ != 'compound'
             and not any(x <= t.idx < y for x, y, _ in spans) and not a <= t.idx < b}
    return before - after


def tail_candidates(doc, text):
    out = []
    for sent in doc.sents:
        for m in TAIL.finditer(sent.text):
            a = sent.start_char + m.start()
            b = sent.end_char
            prefix = text[sent.start_char:a]
            if len(prefix.split()) < 5 or prefix.count('"') % 2 or prefix.count('(') != prefix.count(')'):
                continue
            before = doc.char_span(sent.start_char, a, alignment_mode='contract')
            if not before or not any(t.pos_ in {'VERB', 'AUX'} for t in before):
                continue
            if re.search(r'\b(?:and|or|to|of|for|as|is|are|was|were)\s*$', prefix, re.I):
                continue
            inside = noun_heads(doc, a, b)
            outside = noun_heads(doc, 0, a) | noun_heads(doc, b, len(text))
            out.append(dict(a=a, b=b, cue=m.group('cue').lower(), novel=sorted(inside - outside), words=len(text[a:b].split()), kind='tail'))
            break
    return out


def edit(doc, text, method, guard=None):
    if method == 'previous_best':
        return text, []
    from .ledger_guard import should_protect_span
    spans = []
    cs = tail_candidates(doc, text)
    total = len(text.split())
    used = 0
    has_spec = method.startswith('spec_') or method in {'infer_only', 'strong_hedge', 'likely_hedge'}
    if has_spec:
        for c in cs:
            cue = c['cue']
            if method == 'infer_only' and not re.fullmatch(INFER, cue):
                continue
            if method == 'strong_hedge' and not re.fullmatch(STRONG, cue):
                continue
            if method == 'likely_hedge' and cue != 'likely':
                continue
            if should_protect_span(doc, text, spans, c['a'], c['b'], guard):
                continue
            novel = re.search(r'novel(\d)', method)
            if novel and len(newly_lost_heads(doc, spans, c['a'], c['b'])) > int(novel.group(1)):
                continue
            budget = re.search(r'budget(\d+)', method)
            if budget and (used + c['words']) / max(1, total) > int(budget.group(1)) / 100:
                continue
            used += c['words']
            spans.append((c['a'], c['b'], 'tail_' + cue))
    sentences = list(doc.sents)
    if method.startswith('secondary_') or method == 'spec_novel1_secondary1' or 'background' in method:
        limit = 1 if method == 'spec_novel1_secondary1' else int(method[-1]) if method.startswith('secondary_') else None
        for i, s in enumerate(sentences):
            if i == 0 or not UNCERTAIN.search(s.text):
                continue
            if 'background' in method and not BACKGROUND.search(s.text):
                continue
            a, b = s.start_char, s.end_char
            if should_protect_span(doc, text, spans, a, b, guard):
                continue
            if limit is not None:
                if len(newly_lost_heads(doc, spans, a, b)) > limit:
                    continue
            spans.append((a, b, 'sentence'))
    keep = re.fullmatch(r'first([234])', method)
    if keep or method == 'spec_novel1_first3':
        n = int(keep.group(1)) if keep else 3
        if len(sentences) > n:
            spans.append((sentences[n].start_char, len(text), 'sentence_suffix'))
    if method == 'first_paragraph':
        match = re.search(r'\n\s*\n', text)
        if match and len(text[:match.start()].split()) >= 5:
            spans.append((match.start(), len(text), 'sentence_suffix'))
    result, merged = apply_spans(text, spans)
    assert result
    new_joins = set(re.findall(r'[a-z][.!?][A-Z]', result)) - set(re.findall(r'[a-z][.!?][A-Z]', text))
    assert not new_joins, (method, text, result)
    return result, merged
