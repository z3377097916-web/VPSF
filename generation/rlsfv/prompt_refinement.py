"""Prompt-only views. Never modify the fact ledger, scores or visual verdicts."""
from copy import deepcopy
import re
from .scoring import any_refuted


ABSTRACT_ENTITIES = frozenset({
    'comfort', 'relaxation', 'tranquility', 'atmosphere', 'ambiance',
    'serenity', 'thrill', 'enjoyment', 'peacefulness', 'coziness',
})


def neutral_name(e):
    """Strip only recorded leading qualifiers, preserving plural and compounds."""
    old = e.get('display') or e['anchor']
    if old.lower() in {'black bear', 'brown bear', 'red panda', 'hot dog'}:
        return old
    new = old
    qualifiers = sorted(e.get('qualifiers') or [], key=len, reverse=True)
    while True:
        before = new
        for q in qualifiers:
            new = re.sub(r'^' + re.escape(q) + r'\s+', '', new, flags=re.I)
        if before == new:
            break
    return new or old


def prompt_view(facts):
    view, omitted, renamed = {}, [], []
    for key, original in facts.items():
        f = deepcopy(original)
        ents = f.get('entities') or []
        abstract = [e for e in ents if str(e.get('anchor', '')).lower() in ABSTRACT_ENTITIES]
        if abstract:
            omitted.append({'claim_id': key, 'reason': 'abstract_entity_prompt_only',
                            'entities': [e.get('display') or e['anchor'] for e in abstract]})
            continue
        for e in ents:


            old = e.get('display') or e['anchor']
            new = neutral_name(e)
            if old != new:
                renamed.append({'claim_id': key, 'entity_key': e.get('key'),
                                'before': old, 'after': new})
                e['display'] = new
        view[key] = f
    return view, {'omitted': omitted, 'renamed': renamed,
                  'changes_fact_ledger': False, 'changes_identity_keys': False}


def needs_recheck(f):
    return f.get('modality') == 'definite' and any_refuted(f)
