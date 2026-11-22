"""前瞻动作/不确定身份从句裁剪（E3-V3 新增基于 M1 证据账本的实体与关系保护）。"""
import re, statistics
from .ledger_guard import should_protect_span

IDENTITY = re.compile(r',\s+(?:which|that)\s+(?:(?:appears?|seems?)\s+to\s+be|(?:could|might|may)\s+be)\b', re.I)

FUTURE = re.compile(r'(?P<sep>,\s*|\s+(?:and|or)\s+)(?:(?:possibly|probably|perhaps|likely)\s+)?(?:(?:preparing|getting\s+ready|attempting|trying|planning|intending)\s+to)\b', re.I)


def cuts(nlp, text, variant, doc=None, guard=None):
    doc = nlp(text) if doc is None else doc
    spans = []
    typed_spans = []
    reasons = []
    for sent in doc.sents:
        options = []
        if 'identity' in variant:
            for m in IDENTITY.finditer(sent.text):
                options.append((sent.start_char + m.start(), 'uncertain_relative_identity'))
        if 'prospective' in variant:
            for m in FUTURE.finditer(sent.text):
                options.append((sent.start_char + m.start(), 'future_action_conjunct'))
            for t in sent:
                if t.lower_ != 'to' or t.dep_ != 'aux' or t.head.pos_ != 'VERB':
                    continue
                v = t.head
                if v.dep_ == 'advcl' and v.head.pos_ == 'VERB' and v.head.lemma_ not in {'appear', 'seem', 'look'}:
                    options.append((t.idx, 'purpose_infinitive'))
        for start, reason in sorted(options):
            prefix = text[sent.start_char:start].rstrip()
            if len(prefix.split()) < 5 or prefix.count('"') % 2 or prefix.count('(') != prefix.count(')'):
                continue
            before = doc.char_span(sent.start_char, start, alignment_mode='contract')
            if not before or not any(t.pos_ in {'VERB', 'AUX'} for t in before):
                continue
            if re.search(r'\b(?:and|or|to|of|for|as|is|are|was|were)\s*$', prefix, re.I):
                continue
            # 若裁剪该目的状语/从句会导致丢失受视觉证据支持的具象物体或双支持关系端点，则予以保留
            if should_protect_span(doc, text, typed_spans, start, sent.end_char, guard):
                continue
            spans.append((start, sent.end_char))
            typed_spans.append((start, sent.end_char, reason))
            reasons.append(reason)
            break
    for a, b in reversed(spans):
        suffix = text[b:]
        gap = ' ' if suffix and not suffix[0].isspace() else ''
        text = text[:a].rstrip(' ,;') + '.' + gap + suffix
    return text, spans, reasons
