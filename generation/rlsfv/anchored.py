"""Experimental source-copy extraction; lexical grounding is not entailment.
Separate from formal extraction. Never infer omitted semantic fields.
"""
import json
import re
from .config import CATEGORIES
from .structured import validate_extraction, semantic_warnings
ALIASES=dict(c='category',s='subject',p='predicate',o='object',x='scope',n='polarity',m='modality',t='subtype',e='entities',a='ambiguous')
STRUCTURAL=('exists','is','count','depicts')

def spans(text):
    """All exact contiguous word/punctuation spans, without phrase-length caps."""
    tokens=list(re.finditer(r'\w+|[^\w\s]',text,re.UNICODE))
    return sorted({text[a.start():b.end()] for i,a in enumerate(tokens) for b in tokens[i:]})

def schema(caption,focus):
    full=spans(caption);local=spans(focus)
    def obj(p):return dict(type='object',properties=p,required=list(p),additionalProperties=False)
    def enum(items):return dict(type='string',enum=list(dict.fromkeys(items)))
    return obj({'facts':dict(type='array',items=obj({
        'c':enum(CATEGORIES),'s':enum(full),'p':enum([*STRUCTURAL,*local]),
        'o':enum(['',*local]),'x':enum(['',*local]),
        'n':enum(['positive','negative']),'m':enum(['definite','uncertain']),
        't':dict(type='string'),
        'e':dict(type='array',items=obj({'a':enum(full),'q':dict(type='array',items=enum(full))})),
        'a':dict(type='boolean')}))})

def prompt(caption,focus):
    return '''Extract ALL explicitly stated information in FOCUS, using FULL TEXT only to resolve pronouns. Return {"facts":[...]}.
Use short keys: c category, s subject, p predicate, o object/value, x scope, n polarity (positive/negative), m modality (definite/uncertain), t subtype, e entities [{"a":anchor,"q":[qualifiers]}], a unresolved ambiguity (boolean).
Copy content phrases from the text exactly: s and entity anchors may refer to earlier text; p/o/x must come from FOCUS. Only structural predicates exists/is/count/depicts may be supplied. Use empty o/x/t and empty qualifier lists where unused. t is mandatory for other facts. Entity anchors identify participants, not every adjective or number. Preserve distinct participants and directional roles.
Categories: object existence; attribute property/material/color; count quantity; spatial position; action action/posture; relation interaction/part-whole; text_ocr written text; identity specific identity; scene setting; style medium/style; other remaining explicit claims.
Split independent details (color, material, count, action, relationship) into facts. Preserve the entire phrase for alternatives such as "family or friends"; never turn them into two certain facts. Preserve may/might/appear as uncertain on the claim they qualify. Negative relationships do not prove global nonexistence. Do not invent existence from negated or uncertain clauses. Do not omit subjective claims: record them as source claims, not visually verified truth. Do not repeat a fact. Do not infer gender, material, intention or actions absent from the text.
FULL TEXT: '''+json.dumps(caption,ensure_ascii=False)+'\nFOCUS: '+json.dumps(focus,ensure_ascii=False)

def positions(text,phrase):
    return [[m.start(),m.start()+len(phrase)] for m in re.finditer(re.escape(phrase),text)] if phrase else []

def decode(text,caption,focus):
    """Restore field names, scheduled evidence and IDs; never change meanings."""
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!={'facts'} or not isinstance(value['facts'],list):raise ValueError('ANCHOR_ROOT_REVIEW')
    full=set(spans(caption));local=set(spans(focus));facts=[];provenance=[];ids={}
    for row in value['facts']:
        if not isinstance(row,dict) or set(row)!=set(ALIASES):raise ValueError('ANCHOR_FIELDS_REVIEW')
        if row['s'] not in full or row['p'] not in local|set(STRUCTURAL):raise ValueError('ANCHOR_COPY_REVIEW')
        if row['o'] not in local|{''} or row['x'] not in local|{''}:raise ValueError('ANCHOR_SCOPE_REVIEW')
        f={ALIASES[k]:v for k,v in row.items()};f['quote']=focus;f['entities']=[]
        links={k:positions(caption,row[k]) for k in ['s','p','o','x']}
        for e in row['e']:
            if set(e)!={'a','q'} or e['a'] not in full or not isinstance(e['q'],list) or any(q not in full for q in e['q']):raise ValueError('ENTITY_COPY_REVIEW')
            key=json.dumps([e['a'],e['q']],ensure_ascii=False)
            f['entities'].append(dict(id=ids.setdefault(key,'e'+str(len(ids)+1)),anchor=e['a'],qualifiers=e['q']))
        facts.append(f);provenance.append(links)
    facts=validate_extraction(json.dumps({'facts':facts},ensure_ascii=False),caption,focus)
    return facts,provenance

def audit(caption,focus,facts,nlp=None):
    """Possible-omission diagnostics, not a completeness certificate.
Quotes excluded: a whole-sentence citation must not conceal missing information.
No automatic rewriting, deletion or acceptance follows from these diagnostics.
"""
    warnings=semantic_warnings(facts);values=[]
    for f in facts:
        values.extend(f[k] for k in ['subject','predicate','object','scope'])
        for e in f['entities']:values.extend([e['anchor'],*e['qualifiers']])
    represented=' '.join(values).casefold()
    numbers=r'\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b'
    for m in re.finditer(numbers,focus,re.I):
        if not re.search(r'(?<!\w)'+re.escape(m.group().casefold())+r'(?!\w)',represented):warnings.append(dict(code='COUNT_OMISSION_REVIEW',source=m.group(),offset=[m.start(),m.end()]))
    if nlp is not None:
        represented_lemmas={t.lemma_.casefold() for t in nlp(' '.join(values))}
        for t in nlp(focus):
            if t.pos_ in {'NOUN','PROPN','ADJ','VERB','NUM'} and t.lemma_.casefold() not in represented_lemmas:warnings.append(dict(code='CONTENT_OMISSION_REVIEW',source=t.text,offset=[t.idx,t.idx+len(t.text)]))
    return warnings
