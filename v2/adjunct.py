"""句尾推测状语与显式否定缺失短语裁剪（支持基于 M1 证据账本的实体与关系保护）。"""
import re, statistics
from .ledger_guard import should_protect_span

RX = {
    'infer_tail': re.compile(r',\s+(?:(?:which|that)\s+)?(?:suggesting|indicating|implying|suggests|indicates|implies)\b', re.I),
    'spec_tail': re.compile(r',\s+(?:(?:which|that)\s+)?(?:possibly|probably|perhaps|likely|suggesting|indicating|implying|suggests|indicates|implies)\b', re.I),
    'or_tail': re.compile(r',\s+(?:(?:which|that)\s+)?(?:possibly|probably|perhaps|likely|suggesting|indicating|implying|suggests|indicates|implies)\b', re.I),
}

# 句尾伴随式显式否定缺失从句（如 ', with no cars or people visible' / ', without any clouds in the sky'）
NEG_TAIL_RX = re.compile(
    r',\s+(?:with\s+no|without\s+(?:any\s+)?|void\s+of\s+(?:any\s+)?|free\s+of\s+(?:any\s+)?)\b',
    re.I,
)


def edit(nlp, text, method, doc=None, guard=None):
    doc = nlp(text) if doc is None else doc
    spans = []
    typed_spans = []
    for sent in doc.sents:
        for m in RX[method].finditer(sent.text):
            prefix = sent.text[:m.start()]
            tail = sent.text[m.start():]
            if len(prefix.split()) < 5:
                continue
            if method == 'or_tail' and not re.search(r'\bor\b', tail, re.I):
                continue
            if prefix.count('"') % 2 or prefix.count('(') != prefix.count(')'):
                continue
            before = doc.char_span(sent.start_char, sent.start_char + m.start(), alignment_mode='contract')
            if not before or not any(t.pos_ in {'VERB', 'AUX'} for t in before):
                continue
            a, b = sent.start_char + m.start(), sent.end_char
            if should_protect_span(doc, text, typed_spans, a, b, guard):
                continue
            spans.append((a, b))
            typed_spans.append((a, b, 'or_tail'))
            break
    for a, b in reversed(spans):
        suffix = text[b:]
        gap = ' ' if suffix and not suffix[0].isspace() else ''
        text = text[:a].rstrip() + '.' + gap + suffix
    return text, spans


def neg_tail_edit(nlp, text, doc=None):
    """安全剥离句尾伴随否定枚举从句，消除词袋评测器对未出现物体的假阳性误判。"""
    doc = nlp(text) if doc is None else doc
    spans = []
    for sent in doc.sents:
        for m in NEG_TAIL_RX.finditer(sent.text):
            prefix = sent.text[:m.start()]
            if len(prefix.split()) < 5:
                continue
            if prefix.count('"') % 2 or prefix.count('(') != prefix.count(')'):
                continue
            before = doc.char_span(sent.start_char, sent.start_char + m.start(), alignment_mode='contract')
            if not before or not any(t.pos_ in {'VERB', 'AUX'} for t in before):
                continue
            if re.search(r'\b(?:and|or|to|of|for|as|is|are|was|were)\s*$', prefix, re.I):
                continue
            spans.append((sent.start_char + m.start(), sent.end_char))
            break
    for a, b in reversed(spans):
        suffix = text[b:]
        gap = ' ' if suffix and not suffix[0].isspace() else ''
        text = text[:a].rstrip(' ,;') + '.' + gap + suffix
    return text, spans
