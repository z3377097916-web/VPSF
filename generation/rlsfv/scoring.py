import math
import random
from .config import STAGES, CATEGORIES
from .facts import positive, norm

SUBJECTIVE_SUBTYPES = {"subjective", "atmosphere", "subjective_activity"}


def verified(f):
    """Two completed, schema-valid checks; does NOT mean the fact is true."""
    checks = f.get("checks") or []
    return len(checks) == 2 and all(c.get("status") in {"supported", "refuted", "unclear"} for c in checks)


def effective_statuses(f):
    values = ["unclear" if c.get("low_mass") else c["status"] for c in f["checks"]]
    if values and f.get("soft_support", {}).get("valid") is False:
        values[0] = "unclear"
    return values


def double_refuted(f):
    return verified(f) and effective_statuses(f) == ["refuted", "refuted"]


def any_refuted(f):
    "Either valid check refutes the claim. Low-mass is unclear."
    return verified(f) and "refuted" in effective_statuses(f)


GUARD_RULES = {"any_refuted", "double_refuted"}


DEFAULT_GUARD_RULE = "any_refuted"
FRESH_PRIOR_STRENGTH = 2.0


def guard_rule(cfg):
    """G-constraint ablation rule (H3). Any-refutation is the fix9 default."""
    rule = (cfg or {}).get("guard_rule", DEFAULT_GUARD_RULE)
    if rule not in GUARD_RULES:
        raise ValueError("scoring.guard_rule must be any_refuted|double_refuted")
    return rule


def guard_refuted(f, cfg_or_rule=DEFAULT_GUARD_RULE):
    "Return whether a definite fact is refuted under the selected G rule."
    rule = cfg_or_rule if isinstance(cfg_or_rule, str) else guard_rule(cfg_or_rule)
    if rule not in GUARD_RULES:
        raise ValueError("Unknown guard rule: %s" % (rule,))
    return double_refuted(f) if rule == "double_refuted" else any_refuted(f)


def pseudo_claim(f):
    """抽象/修辞伪命题（非具象 object/identity 或非生命布局惯用伪动作）。延迟导入避免循环依赖。"""
    try:
        from v2.ledger_guard import is_pseudo_claim
    except ImportError:
        return False
    return is_pseudo_claim(f)


def hard_error(f, rule=DEFAULT_GUARD_RULE):
    """计入硬幻觉错误：definite + G 规则反驳，且不是抽象/修辞伪命题。"""
    return f["modality"] == "definite" and guard_refuted(f, rule) and not pseudo_claim(f)


def coverage_eligible(f, cfg):
    "Facts that may enter a coverage numerator/denominator."
    if not positive(f):
        return False
    if cfg.get("exclude_refuted", False) and guard_refuted(f, cfg):
        return False
    if cfg.get("exclude_subjective", False) and f["category"] == "other" and f.get("subtype") in SUBJECTIVE_SUBTYPES:
        return False
    return True
from .io import fingerprint


METHODS = {"B0": {"U": 1}, "Bf": {"U": .5, "f": .5}, "Bs": {"U": .5, "s": .5},
           "Bq": {"U": .5, "q": .5}, "Ball": {"U": .25, "f": .25, "s": .25, "q": .25},


           "BqF": {"U": .5, "qF": .5}}


RLSF10 = "RLSF10"

PRIMARY_GROUP = "G-Bs"
PRIMARY_GROUPS = ("G-B0", "G-Bf", "G-Bs", "G-Bq", "G-Ball", "G-BqF")
ABLATION_GROUPS = ("B0", "Bf", "Bs", "Bq", "Ball", "BqF")

def aggregate_fuzzy(entropies, mask, top_r):
    values = sorted([x for x, use in zip(entropies, mask) if use and x is not None and math.isfinite(x)], reverse=True)
    if not values:
        raise ValueError("INVALID_U: no valid information-token entropy")
    return sum(values[:top_r]) / len(values[:top_r])

def support(verdicts, strict=False):
    if len(verdicts) != 2 or any(v not in {"supported", "refuted", "unclear"} for v in verdicts):
        raise ValueError("Exactly two valid verification results required")
    if "refuted" in verdicts:
        return 0.0
    if strict:
        return float(verdicts == ["supported", "supported"])
    return verdicts.count("supported") / 2

def verified_value(f, cfg, false_support=None):
    "Per-fact verified truth value v(c) in [0, 1]."
    if not verified(f):
        return 0.0
    statuses = effective_statuses(f)
    if "refuted" in statuses:
        return 0.0
    if any(c.get("low_mass") for c in f["checks"]) or f.get("soft_support", {}).get("valid") is False:

        return support(statuses, cfg["strict_q"])
    if not cfg.get("soft_q", False) or "soft_support" not in f:
        return support(statuses, cfg["strict_q"])
    a = f["soft_support"]["supported"]
    if cfg.get("foil_calibration", False) and false_support is not None and false_support < 1:
        a = min(1.0, max(0.0, (a - false_support) / (1 - false_support)))
    b = 1.0 if statuses[1] == "supported" else 0.0
    return (a + b) / 2


def weights(f, cfg, false_support=None):
    fresh_only = cfg.get("frequency_source", "all") == "fresh"
    if fresh_only:
        if "fresh_counts" not in f or "fresh_totals" not in f:
            raise ValueError("Fresh-only scoring requires explicit branch provenance")
        counts, totals = f["fresh_counts"], f["fresh_totals"]
        if list(totals) != [10, 1, 1] or len(counts) != 3 or any(
                type(k) is not int or not 0 <= k <= n for k, n in zip(counts, totals)):
            raise ValueError("Fresh-only scoring requires complete 10/1/1 fresh samples")
    else:
        counts, totals = f["counts"], STAGES
    p = [k/n for k, n in zip(counts, totals)]
    d = sum(x > 0 for x in p) / 3
    # M2: fresh rounds 2/3 have n=1, so raw p_t is only 0 or 1 and J is unstable.

    # with a Beta/Jeffreys-style prior centred on the stable first fresh round.


    prior = (counts[0] + 0.5) / (totals[0] + 1.0) if totals[0] else 0.5
    p_smooth = p if not fresh_only else ([p[0]] + [
        (k + FRESH_PRIOR_STRENGTH * prior) / (n + FRESH_PRIOR_STRENGTH)
        for k, n in zip(counts[1:], totals[1:])])
    if fresh_only:
        # Only adjacent fresh co-occurrences are evidence of stability. A round with
        # zero fresh observations is absence evidence already captured by D, not a

        pairs = [(p_smooth[t], p_smooth[t + 1]) for t in (0, 1)
                 if p[t] > 0 and p[t + 1] > 0]
        raw_pairs = [(p[t], p[t + 1]) for t in (0, 1) if p[t] > 0 and p[t + 1] > 0]
        raw_ratios = [min(a, b)/max(a, b) for a, b in raw_pairs]
        ratios = [min(a, b)/max(a, b) for a, b in pairs]
    else:
        raw_ratios = [min(a, b)/max(a, b) for a, b in zip(p, p[1:]) if max(a,b) > 0]
        ratios = raw_ratios
    j_raw = sum(raw_ratios)/len(raw_ratios) if raw_ratios else 0
    j = sum(ratios)/len(ratios) if ratios else 0
    ws = d * ((1-cfg["eta"]) + cfg["eta"]*j)


    q = support(effective_statuses(f), cfg["strict_q"]) if verified(f) else 0.0
    v = verified_value(f, cfg, false_support)
    frequency = (sum(counts)+cfg["alpha"])/(sum(totals)+2*cfg["alpha"])

    if fresh_only and not sum(counts):
        frequency = 0.0
    return {"f": frequency,
            "s": ws, "q": q*(1+cfg["lambda"]*ws), "support": q, "presence": d,
            "stability": j, "stability_raw": j_raw,
            "fresh_prior_strength": FRESH_PRIOR_STRENGTH if fresh_only else 0.0,
            "v": v, "qv": v*(1+cfg["lambda"]*ws),
            "frequency_source": "fresh" if fresh_only else "all",
            "evidence_counts": list(counts), "evidence_totals": list(totals)}


def branch_provenance(candidates, facts):
    """Rebuild counts from actual candidate membership, never infer branch from order.

    Safe for re-scoring saved records with explicit branches; no legacy imputation.
    """
    seen, totals = set(), {b: [0, 0, 0] for b in ("fresh", "guided")}
    memberships = {k: {b: [0, 0, 0] for b in totals} for k in facts}
    for c in candidates:
        if c["id"] in seen or c.get("branch") not in totals or c["stage"] not in range(3):
            raise ValueError("Invalid or missing candidate branch provenance")
        seen.add(c["id"])
        b, t = c["branch"], c["stage"]
        totals[b][t] += 1
        for k in set(c["claim_ids"]):
            if k not in facts:
                raise ValueError("Candidate references missing fact: " + k)
            memberships[k][b][t] += 1
    if totals != {"fresh": [10, 1, 1], "guided": [0, 6, 4]}:
        raise ValueError("Fresh-only scoring requires the approved 10 + (6+1) + (4+1) design")
    out = {}
    for k, f in facts.items():
        combined = [a+b for a, b in zip(memberships[k]["fresh"], memberships[k]["guided"])]
        if combined != f["counts"]:
            raise ValueError("Fact counts disagree with candidate membership: " + k)
        out[k] = dict(f, fresh_counts=memberships[k]["fresh"], guided_counts=memberships[k]["guided"],
                      fresh_totals=totals["fresh"], guided_totals=totals["guided"])
    return out

def coverage(ids, facts, w, kind, balanced, cfg=None):
    cats = CATEGORIES if balanced else ("all",)
    values, detail = [], {}
    cfg = cfg or {}
    for cat in cats:
        keys = [k for k, f in facts.items() if coverage_eligible(f, cfg) and (cat == "all" or f["category"] == cat)]
        denominator = sum(w[k][kind] for k in keys)
        if denominator > 0:
            v = sum(w[k][kind] for k in keys if k in ids)/denominator
            values.append(v)
            detail[cat] = v
    return (sum(values)/len(values) if values else 0.0), detail

def rank_points(values, ascending=False):
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite ranks are invalid")
    rounded = [round(x, 12) for x in values]
    order = sorted(set(rounded), reverse=not ascending)
    out, previous = {}, 0
    for v in order:
        n = rounded.count(v)
        rank = previous + (n+1)/2
        out[v] = len(values)-rank+1
        previous += n
    return [out[v] for v in rounded]

def verified_precision(ids, facts, w, cfg):
    """Mean v(c) over the candidate's definite, verified, non-subjective claims
    (positive and negative). None when the candidate makes no such claim."""
    keys = [k for k in ids if facts[k]["modality"] == "definite" and verified(facts[k])
            and not (cfg.get("exclude_subjective", False) and facts[k]["category"] == "other"
                     and facts[k].get("subtype") in SUBJECTIVE_SUBTYPES)]
    return (sum(w[k]["v"] for k in keys) / len(keys)) if keys else None


def score(candidates, facts, cfg, seed, calibration=None, evidence_candidates=None):
    if len(candidates) != 22 or [sum(c["stage"] == t for c in candidates) for t in range(3)] != list(STAGES):
        raise ValueError("Scoring requires all 22 captions, including for pool ablations")
    if cfg.get("bs_fallback_rule", "legacy_score") not in {"legacy_score", "min_double_then_any"}:
        raise ValueError("Unknown bs_fallback_rule")
    false_support = (calibration or {}).get("false_support_rate")
    if cfg.get("frequency_source", "all") == "fresh":
        # V2 edits do not become new independent frequency/stability evidence.
        facts = branch_provenance(candidates if evidence_candidates is None else evidence_candidates, facts)
    w = {k: weights(f, cfg, false_support) for k, f in facts.items()}
    primary = cfg.get("coverage_mode", "legacy_categories") == "object_relation"
    if primary:
        from .primary_coverage import build_units, unit_coverage
        units, unit_audit = build_units(facts, w, cfg)

    if cfg["pool"] == 10:
        pool = [c for c in candidates if c["stage"] == 0]
    else:
        start = {22: 0, 12: 1, 5: 2}[cfg["pool"]]
        pool = [c for c in candidates if c["stage"] >= start]
    rows = []
    for c in pool:
        ids = set(c["claim_ids"])
        u = c["fuzzy"]["u"]
        if u is None or not math.isfinite(u):
            raise ValueError("INVALID_U cannot be replaced by zero")
        row = {"id": c["id"], "u": u, "u_star": cfg["rho"]**c["stage"]*u, "coverage": {}, "categories": {},
               "coverage_components": {}, "auxiliary_coverage": {}}
        for kind in ("f", "s", "q", "qv"):
            legacy, row["categories"][kind] = coverage(ids, facts, w, kind, cfg["class_balanced"], cfg)
            if primary:
                row["coverage"][kind], row["coverage_components"][kind] = unit_coverage(ids, units, w, kind)
                row["auxiliary_coverage"][kind] = {
                    k: v for k, v in row["categories"][kind].items() if k not in ("object", "spatial", "relation")}
                row.setdefault("legacy_category_coverage_audit_only", {})[kind] = legacy
            else:
                row["coverage"][kind] = legacy
        p = verified_precision(ids, facts, w, cfg)
        cv = row["coverage"]["qv"]
        row["verified_precision"] = p
        row["coverage"]["qF"] = (2 * p * cv / (p + cv)) if (p is not None and p + cv > 0) else 0.0


        legacy_qv = row.get("legacy_category_coverage_audit_only", {}).get("qv")
        row["legacy_coverage_qv"] = legacy_qv
        row["legacy_qF"] = (2 * p * legacy_qv / (p + legacy_qv)
                            if (p is not None and legacy_qv is not None and p + legacy_qv > 0) else 0.0)
        rule = guard_rule(cfg)
        errors = sorted(k for k in ids if hard_error(facts[k], rule))
        row["errors"] = errors
        row["relation_errors"] = [k for k in errors if facts[k]["category"] in ("spatial", "relation")]
        row["eligible"] = not errors
        row["stage"] = c["stage"]
        row["words"] = len(c["text"].split())
        rows.append(row)
    signals = ("U", "f", "s", "q", "qF")
    ranks = [rank_points([r["u_star"] for r in rows], ascending=True)] + [
        rank_points([r["coverage"][kind] for r in rows]) for kind in signals[1:]]
    for i, r in enumerate(rows):
        r["rank_points"] = dict(zip(signals, [rank[i] for rank in ranks]))
        r["scores"] = {name: sum(a * r["rank_points"][sig] for sig, a in coeff.items()) for name, coeff in METHODS.items()}


    stage0 = [r for r in rows if r["stage"] == 0]
    if stage0:
        u10 = rank_points([r["u"] for r in stage0], ascending=True)
        for r, points in zip(stage0, u10):
            r["rank_points"]["U10"] = points
            r["scores"][RLSF10] = points
    captions = {c["id"]: c for c in pool}
    selections = {}
    for name in METHODS:

        rng = random.Random(int(fingerprint([seed, name])[:16], 16))
        order = rows[:]
        rng.shuffle(order)
        highest = sorted(order, key=lambda r: -round(r["scores"][name], 12))
        lowest = sorted(order, key=lambda r: round(r["scores"][name], 12))
        for guarded in (False, True):
            eligible = [r for r in highest if r["eligible"]] if guarded else highest
            fallback = not eligible
            chosen = (eligible or highest)[0]
            fallback_rule = "legacy_score"
            fallback_original_chosen = chosen["id"]
            fallback_risk = None
            if guarded and fallback and cfg.get("bs_fallback_rule", "legacy_score") == "min_double_then_any":

                def risk_key(row):
                    return (sum(double_refuted(facts[k]) for k in row["errors"]),
                            len(row["errors"]), -round(row["scores"][name], 12))
                chosen = min(order, key=risk_key)
                fallback_rule = "min_double_then_any"
                fallback_risk = list(risk_key(chosen)[:2])
            distinct = [r for r in lowest if norm(captions[r["id"]]["text"]) != norm(captions[chosen["id"]]["text"])]
            if cfg.get("rejected_rule", "lowest") == "length_matched":

                lo, hi = 0.8 * chosen["words"], 1.2 * chosen["words"]
                distinct = [r for r in distinct if lo <= r["words"] <= hi]
            rejected = distinct[0] if distinct else None
            gap = chosen["scores"][name]-rejected["scores"][name] if rejected else 0
            valid = not fallback and rejected is not None and round(gap, 12) > 0
            selections[("G-" if guarded else "")+name] = {
                "chosen": chosen["id"], "rejected": rejected["id"] if rejected else None,
                "tie_seed": int(fingerprint([seed, name])[:16], 16),
                "chosen_ties": [r["id"] for r in (eligible or highest) if round(r["scores"][name],12)==round(chosen["scores"][name],12)
                    and (fallback_rule != "min_double_then_any" or risk_key(r) == risk_key(chosen))],
                "rejected_ties": [r["id"] for r in lowest if rejected and round(r["scores"][name],12)==round(rejected["scores"][name],12)
                    and norm(captions[r["id"]]["text"])!=norm(captions[chosen["id"]]["text"])],
                "fallback_rule": fallback_rule if fallback else None,
                "fallback_original_chosen": fallback_original_chosen if fallback else None,
                "fallback_risk_counts": fallback_risk,
                "score_gap": gap, "pair_valid": valid, "fallback": fallback,
                "chosen_stage": chosen["stage"], "rejected_stage": rejected["stage"] if rejected else None,
                "chosen_words": chosen["words"], "rejected_words": rejected["words"] if rejected else None,
                "guard_rule": ("any_valid_refutation" if guard_rule(cfg) == "any_refuted"
                               else "double_refutation") if guarded else "unguarded",
                "chosen_error_ids": chosen["errors"],
                "skip_reason": "no_eligible" if fallback else "no_distinct_text" if rejected is None else "nonpositive_gap" if not valid else None}


    if stage0:
        rng10 = random.Random(int(fingerprint([seed, RLSF10])[:16], 16))
        order10 = stage0[:]
        rng10.shuffle(order10)
        highest10 = sorted(order10, key=lambda r: -round(r["scores"][RLSF10], 12))
        lowest10 = sorted(order10, key=lambda r: round(r["scores"][RLSF10], 12))
        chosen10 = highest10[0]
        distinct10 = [r for r in lowest10 if norm(captions[r["id"]]["text"]) != norm(captions[chosen10["id"]]["text"])]
        rejected10 = distinct10[0] if distinct10 else None
        gap10 = chosen10["scores"][RLSF10] - rejected10["scores"][RLSF10] if rejected10 else 0
        selections[RLSF10] = {
            "chosen": chosen10["id"], "rejected": rejected10["id"] if rejected10 else None,
            "tie_seed": int(fingerprint([seed, RLSF10])[:16], 16),
            "chosen_ties": [r["id"] for r in highest10 if round(r["scores"][RLSF10], 12) == round(chosen10["scores"][RLSF10], 12)],
            "rejected_ties": [], "score_gap": gap10,
            "pair_valid": False,
            "fallback": False, "chosen_stage": chosen10["stage"], "rejected_stage": rejected10["stage"] if rejected10 else None,
            "chosen_words": chosen10["words"], "rejected_words": rejected10["words"] if rejected10 else None,
            "skip_reason": "paper_method_control"}

    rng = random.Random(int(fingerprint([seed, "RAND", sorted(c["id"] for c in pool)])[:16], 16))
    pick = rng.choice(sorted(rows, key=lambda r: r["id"]))
    selections["RAND"] = {"chosen": pick["id"], "rejected": None, "tie_seed": None, "chosen_ties": [], "rejected_ties": [],
                          "score_gap": 0, "pair_valid": False, "fallback": False, "chosen_stage": pick["stage"],
                          "rejected_stage": None, "chosen_words": pick["words"], "rejected_words": None,
                          "skip_reason": "random_control"}
    return {"config": cfg, "protocol_revision": "fix9.2-bs-review",
            "guard_rule": "any_valid_refutation" if guard_rule(cfg) == "any_refuted" else "double_refutation",
            "fact_weights": w, "rows": rows, "selections": selections,
            "primary_coverage_audit": unit_audit if primary else None,
            "false_support_rate": false_support}
