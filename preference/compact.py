"""Pure offline pairing. No images, LLM calls, reference annotations or re-extraction.

Inputs are the immutable candidate/fact/unit artifacts produced by fix9.1/9.2.
All support here is model support, never a claim of ground-truth correctness.
"""
from collections import Counter
import math
import re

from rlsfv.scoring import effective_statuses, verified, any_refuted, double_refuted, branch_provenance


def features(result):
    if result.get("status") != "complete":
        raise ValueError("Incomplete source result; do not silently omit it")
    rank = result["ranking"]
    if rank["config"].get("guard_rule") != "any_refuted":
        raise ValueError("This protocol requires existing any_refuted G")
    if rank["config"].get("frequency_source") != "fresh":
        raise ValueError("Fresh-only support required")
    units = rank["primary_coverage_audit"]["units"]
    facts = result.get("facts", result.get("ledger", {}).get("facts"))
    if facts is None:
        raise ValueError("Missing saved fact ledger")
    memberships=branch_provenance(result.get('evidence_candidates', result['candidates']),facts)
    if any(rank['fact_weights'][k]['evidence_counts']!=f['fresh_counts'] for k,f in memberships.items()):
        raise ValueError('Saved fresh evidence counts do not match candidate memberships')
    rows = {r["id"]: r for r in rank["rows"]}
    out = []
    for c in result["candidates"]:
        row = rows[c["id"]]
        ids = set(c["claim_ids"])
        if not ids or not ids <= facts.keys() or any(not verified(facts[k]) for k in ids):
            raise ValueError("Missing or incomplete existing fact verification")
        errors = sorted(k for k in ids if facts[k]["modality"] == "definite" and any_refuted(facts[k]))
        if set(errors) != set(row["errors"]) or bool(row["eligible"]) != (not errors):
            raise ValueError("Saved G inconsistent with fact checks")
        present, strong, fresh = {}, {}, {}
        for cat in ("object", "relation"):
            present[cat], strong[cat], fresh[cat] = set(), set(), set()
            for key, unit in units[cat].items():
                members = ids.intersection(unit["claim_ids"])
                if not members:
                    continue
                endpoints = set(unit["endpoint_units"])
                if cat == "relation" and not endpoints <= present["object"]:
                    continue
                present[cat].add(key)
                ss = {k for k in members if effective_statuses(facts[k]) == ["supported"] * 2}
                if ss and (cat == "object" or endpoints <= strong["object"]):
                    strong[cat].add(key)
                    if any(sum(rank["fact_weights"][k]["evidence_counts"]) > 0 for k in ss) and (
                            cat == "object" or endpoints <= fresh["object"]):
                        fresh[cat].add(key)
        gen = c.get("generation", {})
        if gen.get("hit_limit") or gen.get("history_messages", 0):
            raise ValueError("Truncated/history-contaminated description")
        scalar = [row["u_star"], row["scores"]["Bs"]]
        if row['verified_precision'] is not None: scalar.append(row['verified_precision'])
        if not all(math.isfinite(x) for x in scalar):
            raise ValueError("Nonfinite source score")
        out.append({"id": c["id"], "text": c["text"], "stage": c["stage"],
            "branch": c["branch"], "words": len(c["text"].split()),
            "u": row["u_star"], "bs": row["scores"]["Bs"], "p": row["verified_precision"],
            "errors": errors, "double_errors": sorted(k for k in errors if double_refuted(facts[k])),
            "pending": len(c.get("pending_queries", [])), "claim_ids": sorted(ids),
            "primary_claims": sorted(k for k in ids if facts[k]['category'] in {'object','spatial','relation'}
                                     and facts[k]['modality']=='definite' and facts[k]['polarity']=='positive'),
            "double_supported_claims": sorted(k for k in ids if effective_statuses(facts[k])==['supported']*2),
            "positive_supported_claims": sorted(k for k in ids if 'supported' in effective_statuses(facts[k])
                                                and 'refuted' not in effective_statuses(facts[k])),
            "stable_coverage": {g: row["coverage_components"]["s"][g]["coverage"]
                                for g in ("object", "relation")},
            "units": {g: sorted(v) for g, v in present.items()},
            "strong": {g: sorted(v) for g, v in strong.items()},
            "fresh_strong": {g: sorted(v) for g, v in fresh.items()}})
    if len(out) != 22 or Counter(c["stage"] for c in out) != {0: 10, 1: 7, 2: 5}:
        raise ValueError("Expected complete 10+7+5 candidate pool")
    if len({c["id"] for c in out}) != 22:
        raise ValueError("Duplicate candidate id")
    return out

