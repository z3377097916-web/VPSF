"""Conservative replaceable alignment: no embedding or model-inferred equivalence."""
import json
import re
import unicodedata
from .config import CATEGORIES
from .io import fingerprint
from .parse_facts import existence_statement
from .semantics import semantic_payload

def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

class ConservativeAligner:
    def __init__(self, synonyms):
        self.synonyms = {norm(k): norm(v) for k, v in synonyms.items()}
        if any(v in self.synonyms and self.synonyms[v] != v for v in self.synonyms.values()):
            raise ValueError("Synonym map must be one-step canonical, not chained")

    def value(self, text):
        v = norm(text)
        return self.synonyms.get(v, v)

    def canonical(self, fact, candidate_id):
        payload = {k: self.value(fact[k]) for k in
                   ("category", "subject", "predicate", "object", "scope", "polarity", "modality", "subtype")}

        payload["entities"] = sorted([{
            "anchor": self.value(e["anchor"]), "qualifiers": sorted(map(self.value, e["qualifiers"]))
        } for e in fact["entities"]], key=lambda e: json.dumps(e, sort_keys=True))
        if fact["ambiguous"]:
            payload["unresolved_caption"] = candidate_id
        return fingerprint(payload)

class LemmaAligner(ConservativeAligner):
    "Align on (category, subject head, predicate, object head, polarity, modality)."
    def key_part(self, text):
        parts = []
        for alt in str(text).split("|"):
            head, bracket, rest = alt.partition("[")
            parts.append(self.value(head) + (bracket + rest if bracket else ""))
        return "|".join(sorted(parts))

    def canonical(self, fact, candidate_id):
        align = fact.get("align")
        if align is None:
            ents = fact.get("entities") or []
            if fact["category"] == "object" and ents:
                subject = ents[0].get("key") or ents[0].get("head") or ents[0]["anchor"]
            else:
                subject = fact["subject"]
            align = {"subject": subject, "predicate": fact["predicate"], "object": fact["object"]}
        payload = semantic_payload(dict(fact, align=align), self.key_part)
        if fact["ambiguous"]:
            payload["unresolved_caption"] = candidate_id
        return fingerprint(payload)


def make_aligner(cfg):
    mode = cfg.get("alignment", {}).get("mode", "conservative")
    if mode == "lemma":
        return LemmaAligner(cfg["synonyms"])
    if mode == "conservative":
        return ConservativeAligner(cfg["synonyms"])
    raise ValueError("Unknown alignment mode")


def parse_facts(text, caption):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"facts"} or not isinstance(value["facts"], list):
        raise ValueError("Extraction must return {facts: [...]} only")
    fields = {"category", "subject", "predicate", "object", "scope", "polarity", "modality",
              "subtype", "quote", "entities", "ambiguous"}
    entity_anchors = {}
    for f in value["facts"]:
        if not isinstance(f, dict) or set(f) != fields:
            raise ValueError("Fact schema mismatch")
        if any(not isinstance(f[k], str) for k in fields - {"entities", "ambiguous"}):
            raise ValueError("Fact text fields must be strings")
        if f["category"] not in CATEGORIES or f["polarity"] not in ("positive", "negative") or f["modality"] not in ("definite", "uncertain"):
            raise ValueError("Unknown fact category/polarity/modality")
        if not f["quote"] or f["quote"] not in caption:
            raise ValueError("Fact evidence must be an exact caption span")
        if not f["subject"] or not f["predicate"] or type(f["ambiguous"]) is not bool:
            raise ValueError("Missing subject/predicate/ambiguity")
        if f["category"] == "other" and not f["subtype"]:
            raise ValueError("Other facts need a concrete subtype")
        if not isinstance(f["entities"], list):
            raise ValueError("Entities must be a list")
        local = set()
        for e in f["entities"]:
            if not isinstance(e, dict) or set(e) != {"id", "anchor", "qualifiers"}:
                raise ValueError("Entity schema mismatch")
            if not isinstance(e["id"], str) or not e["id"] or not isinstance(e["anchor"], str) or not e["anchor"] or e["id"] in local:
                raise ValueError("Invalid entity anchor or duplicate local ID")
            if not isinstance(e["qualifiers"], list) or any(not isinstance(x, str) for x in e["qualifiers"]):
                raise ValueError("Invalid entity qualifiers")
            prior = entity_anchors.setdefault(e["id"], norm(e["anchor"]))
            if prior != norm(e["anchor"]):
                raise ValueError("ENTITY_ID_REVIEW: one local ID has conflicting entity anchors")
            local.add(e["id"])
    return value["facts"]

def union(candidates, aligner):
    result = {}

    known_branches = all(c.get("branch") in {"fresh", "guided"} for c in candidates)
    totals = {b: [sum(c["stage"] == t and c.get("branch") == b for c in candidates)
                  for t in range(3)] for b in ("fresh", "guided")}
    for c in candidates:
        ids = set()
        for f in c["facts"]:
            key = aligner.canonical(f, c["id"])
            ids.add(key)
            if key not in result:
                result[key] = dict(f, id=key, counts=[0, 0, 0], candidates=[], quotes=[])
                if known_branches:
                    result[key].update(fresh_counts=[0, 0, 0], guided_counts=[0, 0, 0],
                                       fresh_totals=totals["fresh"][:], guided_totals=totals["guided"][:])
            result[key]["quotes"].append({"candidate": c["id"], "quote": f["quote"]})
        c["claim_ids"] = sorted(ids)
        for key in ids:
            result[key]["counts"][c["stage"]] += 1
            result[key]["candidates"].append(c["id"])
            if known_branches:
                result[key][c["branch"] + "_counts"][c["stage"]] += 1
    for f in result.values():
        f["first_seen"] = next(i for i, k in enumerate(f["counts"]) if k)
        f["presence"] = [bool(k) for k in f["counts"]]
    return result

def positive(f):
    return f["polarity"] == "positive" and f["modality"] == "definite"

def verification(text, template=0):
    return parse_verification(text, template)[0]

def parse_verification(text, template=0):
    """(verdict, format_normalized)。规格 7.4 / 15.3 R1（2026-10-24 决定允许格式规范化）：
    原文 json.loads 失败时，只去掉 ``` 代码块标记、取第一个完整的 {...} 对象再解析；
    之后的键集合与 status 检查完全不变。规范化后仍不合法照样报错，绝不当作 unclear。"""
    normalized = False
    try:
        v = json.loads(text)
    except ValueError:
        body = re.sub(r"```(?:json|JSON)?", "", text)
        start = body.find("{")
        if start < 0:
            raise
        v, _ = json.JSONDecoder().raw_decode(body[start:])
        normalized = True
    if not isinstance(v, dict) or set(v) != ({"status", "evidence"} if template == 0 else {"observation", "status", "evidence"}):
        raise ValueError("Verification schema mismatch")
    if v["status"] not in ("supported", "refuted", "unclear") or not isinstance(v["evidence"], str):
        raise ValueError("Invalid verification, not equivalent to unclear")
    if template == 1 and not isinstance(v["observation"], str):
        raise ValueError("Template B requires an observation")
    return v, normalized

def caption_valid(text):
    s = text.strip()
    if not s or s[0] in "{[" or re.fullmatch(r"(?i)(yes|no|supported|refuted|unclear)[.!]?", s):
        raise ValueError("FORMAT_REVIEW: response is not a complete caption")
    if re.search(r"(?m)^\s*(?:[-*] |\d+[.)] )", s) or "```" in s:
        raise ValueError("FORMAT_REVIEW: checklist/code output requires review")
    return s

def ensure_existence(facts):
    """Only existence logically entailed by a definite positive entity claim.
    An uncertain/negated claim never creates a positive existence fact.
    Canonical per-caption deduplication subsequently removes duplicates.
    """
    result=list(facts)
    for f in facts:
        if not positive(f) or f["category"]=="object": continue

        if f.get("alternatives"): continue
        for e in f["entities"]:
            if e.get("alternative"): continue
            fact = {"category":"object","subject":e["anchor"],"predicate":"exists","object":"",
                "scope":"" if "align" in f else f["scope"],"polarity":"positive","modality":"definite","subtype":"",
                "quote":f["quote"],"entities":[e],"ambiguous":f["ambiguous"],
                "derivation":"entity existence directly entailed by definite positive claim"}


            if "key" in e:
                fact["align"] = {"subject": e["key"], "predicate": "exists", "object": ""}
                fact["statement"] = existence_statement(e.get("display") or e["anchor"], bool(e.get("plural")))
            result.append(fact)
    return result
