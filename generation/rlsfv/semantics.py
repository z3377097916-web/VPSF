"""Shared conservative identity for within-caption and cross-caption facts."""
import re


def normalize(text):
    return " ".join(str(text or "").casefold().split())


def semantic_payload(fact, value=normalize):
    align = fact.get("align") or {k: fact.get(k, "") for k in ("subject", "predicate", "object")}
    def part(text):
        return "|".join(sorted(value(x) for x in str(text or "").split("|")))
    result = {"category": fact["category"], "subject": part(align.get("subject")),
              "predicate": value(align.get("predicate") or ""), "object": part(align.get("object")),
              "polarity": fact["polarity"], "modality": fact["modality"]}


    if fact.get("scope"):
        result["scope"] = value(fact["scope"])
    if fact["category"] in {"spatial", "relation"}:
        roles = fact.get("semantic_roles") or {}
        if roles:
            result["roles"] = {k: sorted(part(x) for x in v) for k, v in sorted(roles.items()) if v}
        else:

            extras = (fact.get("entities") or [])[2:]
            if extras:
                result["extra_participants"] = sorted(part(e.get("key") or e["anchor"]) for e in extras)
    if fact["category"] == "other":
        result["subtype"] = value(fact.get("subtype", ""))
    return result


def unary_location(fact):
    """Only explicit absolute-location parses or grounded location phrases qualify."""
    if fact.get("category") != "spatial" or len(fact.get("entities") or []) != 1:
        return False
    if fact.get("relation_arity") == "unary_location":
        return not re.search(r"\b(?:of|than)\s*$", fact.get("predicate", ""), re.I)
    text = " ".join(str(fact.get(k, "")) for k in ("predicate", "object"))
    return bool(re.search(r"\b(?:in|at|on) (?:the )?(?:background|foreground|center|centre|middle|distance)\b",
                          text, re.I)) and not re.search(r"\bof\b", text, re.I)
