"""Narrow, source-preserving guard for mechanically incomplete fact sentences.

This is not another extractor or a visual judge. Unknown or unusual complete
claims are untouched. Pending claims cannot seed derived existence claims.
"""
import re
from copy import deepcopy


def issues(fact):


    if 'statement' not in fact:
        return []
    s = str(fact.get('statement') or '').strip()
    body = s.rstrip('.!? \t\n')
    if not body:
        return ['empty_statement']

    if re.search(r"\s['’](?:s|re|m|ve|ll|d)$", body, flags=re.I):
        return ['dangling_contraction_predicate']

    if re.fullmatch(r'There\s+(?:is|are|was|were)', body, flags=re.I):
        return ['existential_without_entity']
    if re.match(r'^The\s+\S', body, flags=re.I) and re.search(
            r'\s(?:is|are|was|were|has|have|had)$', body, flags=re.I):
        return ['missing_predicate_complement']
    return []


def apply_candidate(candidate):
    accepted, pending = [], []
    for original in candidate['facts']:
        reasons = issues(original)
        if not reasons:
            accepted.append(original)
            continue
        f = deepcopy(original)
        f['statement_guard'] = {'status': 'pending', 'reasons': reasons,
                                'original_statement': f.get('statement'), 'original_quote': f.get('quote')}
        pending.append(f)
        candidate.setdefault('pending_queries', []).append({
            'quote': f.get('quote', ''), 'focus': f.get('statement', ''),
            'anchors': [e['anchor'] for e in f.get('entities', [])],
            'issue': ';'.join(reasons), 'kind': 'statement_integrity',
            'eligible_for_scoring': False, 'extraction_status': 'unresolved', 'visual': True})
    candidate['facts'] = accepted
    candidate['statement_guard'] = {'version': '10.1', 'pending_count': len(pending),
                                     'pending_facts': pending}
    return candidate
