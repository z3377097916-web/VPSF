"Internal, model-supported object/edge coverage. NOT a benchmark recall metric."
from .facts import norm, positive
from .io import fingerprint
from .semantics import semantic_payload, unary_location


def entity_identity(e):
    return fingerprint({"key": norm(e.get("key") or e.get("head") or e["anchor"]),
                        "qualifiers": sorted(norm(x) for x in e.get("qualifiers", []))})


def build_units(facts, weights, cfg):

    from .scoring import coverage_eligible
    units = {"object": {}, "relation": {}}
    rejected = {}

    def admissible(k, f):
        return (positive(f) and not f.get("ambiguous", False)
                and coverage_eligible(f, cfg) and weights[k]["support"] > 0
                and weights[k]["v"] > 0)

    for k, f in facts.items():
        if f["category"] != "object":
            continue
        ents = f.get("entities") or []
        if not admissible(k, f):
            rejected[k] = "object_not_visually_supported_or_ambiguous"
            continue
        if len(ents) != 1 or ents[0].get("alternative") or "|" in str(ents[0].get("key", "")):
            rejected[k] = "object_identity_unresolved"
            continue
        key = entity_identity(ents[0])
        units["object"].setdefault(key, {"claim_ids": [], "endpoint_units": []})["claim_ids"].append(k)

    for k, f in facts.items():
        if f["category"] not in {"spatial", "relation"}:
            continue
        if not admissible(k, f):
            rejected[k] = "relation_not_visually_supported_or_ambiguous"
            continue
        if unary_location(f):
            rejected[k] = "unary_location_not_a_relation_edge"
            continue
        ents = f.get("entities") or []
        if any(e.get("alternative") or "|" in str(e.get("key", "")) for e in ents):
            rejected[k] = "relation_endpoint_unresolved"
            continue
        endpoints = sorted({entity_identity(e) for e in ents})
        if len(endpoints) < 2 or any(e not in units["object"] for e in endpoints):
            rejected[k] = "relation_endpoints_not_separately_supported"
            continue


        key = fingerprint({"claim": semantic_payload(f), "endpoints": endpoints})
        unit = units["relation"].setdefault(key, {"claim_ids": [], "endpoint_units": endpoints})
        unit["claim_ids"].append(k)
    for group in units.values():
        for unit in group.values():
            unit["claim_ids"].sort()
    audit = {"definition": "model_supported_entity_and_directed_edge_proxy_not_ground_truth",
             "group_weights": {"object": 0.5, "relation": 0.5},
             "absent_group_rule": "renormalize_over_positive_denominators",
             "duplicate_unit_weight": "maximum_member_weight_not_sum",
             "units": units, "excluded_claims": rejected,
             "unit_counts": {g: len(items) for g, items in units.items()}}
    return units, audit


def unit_coverage(ids, units, weights, kind):
    details = {}
    ids = set(ids)


    present_objects = {k for k, unit in units["object"].items()
                       if ids.intersection(unit["claim_ids"])
                       and any(weights[cid][kind] > 0 for cid in unit["claim_ids"])}
    values = []
    for group in ("object", "relation"):
        denominator = numerator = 0.0
        matched = total = 0
        for key, unit in units[group].items():


            weight = max(weights[k][kind] for k in unit["claim_ids"])
            if weight <= 0:
                continue
            total += 1
            denominator += weight
            present = bool(ids.intersection(unit["claim_ids"]))
            if group == "relation":
                present = present and all(e in present_objects for e in unit["endpoint_units"])
            if present:
                numerator += weight
                matched += 1
        value = numerator / denominator if denominator else None
        details[group] = {"coverage": value, "numerator": numerator, "denominator": denominator,
                          "matched_units": matched, "total_units": total,
                          "status": "scored" if denominator else "no_positive_weight_units"}
        if value is not None:
            values.append(value)

    return (sum(values) / len(values) if values else 0.0), details
