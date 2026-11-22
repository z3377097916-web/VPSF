"""盲标析取视觉择优规则（保留 V2 二元择优契约，并新增多路 A, B, or C 视觉择优替换）。"""
import re, statistics


def decision(group, method):
    if method in {'first', 'last'}:
        return method
    if method.startswith('context_choice'):
        choices = []
        for check in group['choice_checks']:
            raw = check['raw']
            letter = max(raw['probs'], key=raw['probs'].get)
            if raw['mass'] < .3 or (method == 'context_choice' and raw['probs'][letter] < .65):
                return None
            choices.append(check['mapping'][letter])
        return choices[0] if choices[0] == choices[1] and choices[0] in {'first', 'last'} else None
    if method in {'g_argmax', 'g_margin', 'g_agree', 'g_entropy_agree'}:
        ps = [[x['p_yes'] for x in side] for side in group['checks']]
        if any(x['low_mass'] for side in group['checks'] for x in side):
            return None
        avg = list(map(statistics.mean, ps))
        diff = avg[0] - avg[1]
        if max(avg) < .65 or abs(diff) <= 1e-6:
            return None
        winner = 'first' if diff > 0 else 'last'
        if method == 'g_margin' and abs(diff) < .10:
            return None
        if method == 'g_agree' and not all((a - b) * diff > 0 for a, b in zip(*ps)):
            return None
        if method == 'g_entropy_agree' and decision(group, 'entropy_head') != winner:
            return None
        return winner
    statuses = [[x['status'] for x in side] for side in group['checks']]
    a_supported = statuses[0] == ['supported'] * 2
    b_supported = statuses[1] == ['supported'] * 2
    a_rejected = statuses[0] == ['refuted'] * 2
    b_rejected = statuses[1] == ['refuted'] * 2
    strict = 'first' if a_supported and b_rejected else 'last' if b_supported and a_rejected else None
    if method == 'g_strict':
        return strict
    if method == 'g_supported':
        return 'first' if a_supported and not b_supported else 'last' if b_supported and not a_supported else None
    if method == 'g_strict_then_entropy':
        if strict:
            return strict
        method = 'entropy_head'
    metric = method.removeprefix('entropy_')
    a = group['entropy']['a'][metric]
    b = group['entropy']['b'][metric]
    if a is None or b is None or abs(a - b) <= 1e-6:
        return None
    return 'first' if a < b else 'last'


def apply(text, groups, method):
    spans = []
    decisions = []
    for group in groups:
        side = decision(group, method)
        decisions.append(side)
        if side is None:
            continue
        start, end = group[side]
        if any(not (end <= a or start >= b) for a, b in spans):
            raise ValueError('Overlapping edits')
        spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        left, right = text[:start], text[end:]
        if right[:1] in {',', ';', '.', ':', '!', '?'}:
            left = left.rstrip()
        text = left + right
    text = re.sub(' +', ' ', text).strip()
    return text, decisions


def multiway_decision(mgroup):
    """在三元及以上并列项中选出唯一最高视觉支持度的优胜项索引（低质量或平局时弃权）。"""
    checks = mgroup.get('checks') or []
    if not checks:
        return None
    if any(x.get('low_mass') for opt_checks in checks for x in opt_checks):
        return None
    avgs = [statistics.mean(x['p_yes'] for x in opt_checks) for opt_checks in checks]
    ranked = sorted(enumerate(avgs), key=lambda kv: kv[1], reverse=True)
    best_idx, best_val = ranked[0]
    second_val = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_val < 0.65 or (best_val - second_val) <= 1e-6:
        return None
    winner_text = mgroup['options'][best_idx]['text']
    if re.search(r'\b(?:similar|same|other|another|such)\b', winner_text, re.I):
        return None
    return best_idx


def apply_multiway(text, mgroups):
    """对无重叠的多元名词并列组执行安全单选替换。"""
    spans = [tuple(g['full_span']) for g in mgroups]
    ambiguous = {
        i for i, (a, b) in enumerate(spans)
        if any(j != i and max(a, c) < min(b, d) for j, (c, d) in enumerate(spans))
    }
    replacements = []
    decisions = []
    for i, g in enumerate(mgroups):
        if i in ambiguous:
            g['abstention_reason'] = 'ambiguous_overlapping_multiway'
            decisions.append(None)
            continue
        idx = multiway_decision(g)
        decisions.append(idx)
        if idx is not None:
            start, end = g['full_span']
            replacements.append((start, end, g['options'][idx]['replacement']))
    for start, end, rep in sorted(replacements, reverse=True):
        left, right = text[:start], text[end:]
        if right[:1] in {',', ';', '.', ':', '!', '?'}:
            rep = rep.rstrip()
        text = left + rep + right
    text = re.sub(' +', ' ', text).strip()
    return text, decisions
