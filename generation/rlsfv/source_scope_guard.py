"Separately enabled, conservative source-fidelity guard."
import re
from copy import deepcopy


def issues(f):
    quote=(f.get('quote') or '').lower()
    statement=(f.get('statement') or '').lower()
    out=[]
    obj=str(f.get('object') or '').lower()


    with_location=bool(obj and re.search(
        r'\bwith\s+(?:(?:a|an|the|some|several)\s+)?(?:\w+\s+){0,2}'
        +re.escape(obj)+r'\s+(?:(?:is|are|being)\s+)?(?:closer|further|farther|towards?|behind)\b',quote))
    if (f.get('category')=='relation' and f.get('predicate')=='with'
            and f.get('rule')=='prep_with' and re.search(r'\bhas\b',statement)
            and with_location):

        # not by itself license the renderer's ownership/possession 'has'.
        out.append('with_does_not_resolve_possession_or_attachment')
    head=str((f.get('entities') or [{}])[0].get('head') or f.get('subject') or '').lower()
    same_type_other=bool(head and re.search(r'\b(?:another|other)\s+(?:(?:one|two|three|four)\s+)?'+re.escape(head)+r's?\b',quote))
    pronominal_other=bool(re.search(r'\b(?:others|third one)\b|\banother\s+(?:is|was|stands|sits|lies|one\b|on\b)',quote))
    if (f.get('category') in {'object','count'}
            and re.search(r'^there (?:is|are) (?:one|1)\b',statement)
            and re.search(r'\b(?:one|1)\b',quote)
            and (same_type_other or pronominal_other)
            and not re.search(r'\b(?:only|exactly)\s+(?:one|1)\b',quote)):
        out.append('subset_one_must_not_be_global_count')
    return out


def separate(facts):
    accepted,pending=[],[]
    for original in facts:
        reasons=issues(original)
        if reasons:
            f=deepcopy(original)
            f['source_scope_guard']={'status':'pending','reasons':reasons,
                                     'original_statement':original.get('statement'),
                                     'original_quote':original.get('quote')}
            pending.append(f)
        else:
            accepted.append(original)
    return accepted,pending


def apply_candidate(candidate):
    """Runs after ensure_existence so inferred object facts cannot bypass it."""
    accepted,pending=separate(candidate['facts'])
    candidate['facts']=accepted
    candidate['source_scope_guard']={'version':'fix9.2-scope-trial',
                                    'pending_facts':pending,'pending_count':len(pending)}
    for f in pending:
        candidate.setdefault('pending_queries',[]).append({
            'quote':f.get('quote',''),'focus':f.get('statement','').rstrip('.'),
            'issue':';'.join(f['source_scope_guard']['reasons']),
            'anchors':[e['anchor'] for e in f.get('entities',[])],
            'kind':'source_scope_guard','eligible_for_scoring':False,
            'extraction_status':'unresolved','visual':True})
    return candidate
