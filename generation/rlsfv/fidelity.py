"Extraction-fidelity audit: does each structured fact remain faithful to its quoted caption?"
from __future__ import annotations

import re
from .coref import grounded_antecedents
from .semantics import unary_location

NEG = re.compile(r"\b(not|no|none|never|without|cannot|can't|isn't|aren't|doesn't|don't|n't)\b", re.I)

UNC = re.compile(r"\b(may|might|possibly|perhaps|appear|appears|seem|seems|look|looks|likely|probably|could|"
                 r"or|another|some|several|many)\b", re.I)
PASSIVE = re.compile(r"\b(?:is|are|was|were|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|held|worn|seen|taken|given|ridden|drawn|driven)\b[^.]{0,40}\bby\b", re.I)
COLORS = set("red blue green yellow purple orange pink white black brown gray grey colorful golden silver".split())
DIRECTIONAL = ("above", "below", "left", "right", "front", "behind", "on top", "under", "inside",
               "over", "beneath", "underneath", "side of", "next to", "across", "against",
               "in the middle", "between")
SYMMETRIC = ("next to", "beside", "near", "between", "with", "along", "around")
PRONOUNS = {"it", "they", "them", "its", "their", "another", "others", "one", "ones", "this", "that"}

CRITICAL = frozenset({
    "empty_quote", "object_not_in_quote", "negative_without_cue", "uncertain_without_cue",
    "colour_far_from_anchor", "relation_missing_counterpart", "relation_anchor_not_in_quote",
    "relation_direction_unverified", "relation_passive_unverified",
})


def _tokens(text):
    return re.findall(r"[a-z']+", (text or "").lower())


def _word_in_quote(word, quote_lower):
    if not word:
        return True
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", quote_lower) is not None


def _is_directional(predicate):
    pred = (predicate or "").lower()
    if any(s in pred for s in SYMMETRIC):
        return False
    return any(re.search(r"\b" + re.escape(d) + r"\b", pred) for d in DIRECTIONAL)


def audit_fact(f, caption=None):
    """Return (issues, detail) for one fact. ``caption`` is optional context."""
    issues, detail = [], {}
    quote = f.get("quote") or ""
    ql = quote.lower()
    if not quote.strip():
        issues.append("empty_quote")
        return issues, detail
    cat = f.get("category", "?")
    subject = (f.get("subject") or "").lower()
    obj = (f.get("object") or "").lower()
    pred = (f.get("predicate") or "").lower()
    licensed = grounded_antecedents(f, caption)

    if subject and not _word_in_quote(subject, ql) and subject not in PRONOUNS:
        detail["subject"] = subject
        issues.append("subject_resolved_outside_quote")
    if obj and cat in {"attribute", "count", "spatial", "relation", "identity", "text_ocr"} \
            and not _word_in_quote(obj, ql) and obj not in licensed:
        issues.append("object_not_in_quote")
    pol, mod = f.get("polarity"), f.get("modality")
    has_neg = bool(NEG.search(ql))
    has_unc = bool(UNC.search(ql))
    if pol == "negative" and not has_neg:
        issues.append("negative_without_cue")
    if pol == "positive" and has_neg and re.search(r"\b(not|no|never|without|none)\b", ql):
        issues.append("positive_with_negation_cue")
    if mod == "uncertain" and not has_unc:
        issues.append("uncertain_without_cue")

    if cat == "attribute" and obj in COLORS and subject:
        qt = _tokens(quote)
        if subject in qt:
            i = qt.index(subject)
            dists = [abs(j - i) for j, t in enumerate(qt) if t == obj]
            if dists and min(dists) > 6:
                detail["colour_anchor_distance"] = min(dists)
                issues.append("colour_far_from_anchor")

    if cat in {"spatial", "relation"}:
        ents = f.get("entities") or []
        anchors = [(e.get("anchor") or "").lower() for e in ents]
        unary = unary_location(f)
        if len(ents) < 2 and not unary:
            issues.append("relation_missing_counterpart")
        else:
            missing = [a for a in anchors if a and a not in PRONOUNS and not _word_in_quote(a, ql) and a not in licensed]
            if missing:
                detail["anchors_not_in_quote"] = missing
                issues.append("relation_anchor_not_in_quote")
        directional = _is_directional(pred) and not unary
        checks = (f.get("adjudication") or {}).get("checks") or []
        role_checked = any(c.get("kind") == "role" or c.get("kind", "").startswith("direction_") for c in checks)
        if directional and not role_checked:
            issues.append("relation_direction_unverified")
        if not f.get("statement_reversed"):
            issues.append("relation_no_reversed")
        if f.get("voice") == "passive" or PASSIVE.search(f.get("statement", "")):
            detail["passive_fact"] = True
            issues.append("relation_passive_voice")
            if not role_checked:
                issues.append("relation_passive_unverified")
        if len(ents) >= 2 and all(a in PRONOUNS for a in anchors if a):
            issues.append("relation_pronoun_only")
    return issues, detail


def audit_facts(facts):
    """Audit a mapping/list of facts; returns {fact_id: (issues, detail)}."""
    out = {}
    if isinstance(facts, dict):
        items = facts.values()
    else:
        items = facts
    for f in items:
        out[f.get("id", id(f))] = audit_fact(f)
    return out


def apply_audit(facts, caption=None):
    """Attach fidelity_issues/fidelity_detail. Returns the list of fact ids that have critical issues."""
    critical = []
    for f in facts:
        issues, detail = audit_fact(f, caption)
        if issues:
            f["fidelity_issues"] = issues
        if detail:
            f["fidelity_detail"] = detail
        if any(i in CRITICAL for i in issues):
            critical.append(f.get("id"))
    return critical
