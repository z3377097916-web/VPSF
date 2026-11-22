"Hybrid caption -> facts extraction: deterministic parser proposes, frozen LLaVA adjudicates."
from __future__ import annotations

import time

from .adjudicate import Adjudicator
from .fidelity import apply_audit
from .coref import resolve as resolve_pronouns
from .io import fingerprint
from .parse_facts import parse_caption
from . import lexicon as L

DOWNSTREAM = ("category", "subject", "predicate", "object", "scope", "polarity", "modality",
              "subtype", "quote", "entities", "ambiguous")


def validate_fact(f, caption):
    """Structural checks only (the same contract as facts.parse_facts, plus audit fields)."""
    from .config import CATEGORIES
    for k in DOWNSTREAM:
        if k not in f:
            raise ValueError("HYBRID_SCHEMA: missing " + k)
    if f["category"] not in CATEGORIES or f["polarity"] not in ("positive", "negative") \
            or f["modality"] not in ("definite", "uncertain"):
        raise ValueError("HYBRID_SCHEMA: bad label")
    if not f["quote"] or f["quote"] not in caption:
        raise ValueError("HYBRID_SOURCE: quote is not a caption span")
    for a, b in f.get("spans", []):
        if not (0 <= a < b <= len(caption)):
            raise ValueError("HYBRID_SOURCE: span outside caption")
    if f["category"] == "other" and not f["subtype"]:
        raise ValueError("HYBRID_SCHEMA: other needs subtype")
    return f


def pending_query(item, kind):
    """Clue for the next round's question list; never scored (eligible_for_scoring=False)."""
    return {"quote": item["quote"], "issue": item["issue"], "anchors": item.get("anchors", []),
            "focus": item.get("focus", ""), "visual": item.get("visual", True),
            "align_key": item.get("align_key"),
            "kind": kind, "eligible_for_scoring": False, "extraction_status": "unresolved"}


class HybridExtractor:
    def __init__(self, backend, store, nlp, settings=None):
        s = dict(settings or {})
        self.nlp, self.store = nlp, store
        self.fidelity_audit = bool(s.get("fidelity_audit", False))
        self.pronoun_resolution = bool(s.get("pronoun_resolution", False))
        self.direction_consistency = bool(s.get("direction_consistency", False))
        self.adjudicator = Adjudicator(backend, store, mode=s.get("adjudicate", "risky"),
                                       reject_threshold=s.get("reject_threshold", 0.6),
                                       min_mass=s.get("min_option_mass", 0.3),
                                       calibrate=s.get("calibrate", False),
                                       claim_check=s.get("claim_check", False), seed=s.get("seed", 42),
                                       direction_consistency=self.direction_consistency)

    def extract(self, caption):
        started = time.monotonic()
        parsed = parse_caption(self.nlp, caption)
        sentences = parsed["sentences"]
        if self.pronoun_resolution:

            parsed["facts"] = resolve_pronouns(parsed["facts"], sentences)
        accepted, pending_facts, rejected = [], [], []
        self.adjudicator.prefetch(parsed["facts"], sentences)
        fidelity_pending = 0
        for f in parsed["facts"]:
            validate_fact(f, caption)
            status, record = self.adjudicator.judge(f, sentences)
            f["adjudication"] = dict(record, status=status)
            if self.fidelity_audit:
                critical = apply_audit([f], caption)
                if critical and status == "accepted":


                    status = "pending"
                    f["adjudication"]["status"] = "pending"
                    f["adjudication"]["reason"] = "fidelity: " + ",".join(f.get("fidelity_issues", []))
                    fidelity_pending += 1
            (accepted if status == "accepted" else pending_facts if status == "pending" else rejected).append(f)
        queries = []
        for f in pending_facts:
            queries.append(pending_query({"quote": f["quote"], "issue": "unsettled claim: " + f["statement"]
                                          + " (" + f["adjudication"].get("reason", "") + ")",
                                          "anchors": [e["anchor"] for e in f["entities"]],
                                          "focus": f["statement"].rstrip("."),
                                          "align_key": list(f["align"].values()) + [f["category"]]}, "pending_fact"))
        for p in parsed["pending"]:
            token = p["issue"].split("'")[1] if "'" in p["issue"] else ""
            visual = token.lower() not in L.SUBJECTIVE
            queries.append(pending_query(dict(p, focus=p.get("focus") or self.focus(caption, p), visual=visual), "parser"))
        audit = {
            "caption_hash": fingerprint(caption), "method": "hybrid_parser_adjudicated",
            "proposed": len(parsed["facts"]), "accepted": len(accepted), "pending": len(pending_facts),
            "rejected": len(rejected), "uncovered_words": len(parsed["uncovered"]),
            "content_coverage": parsed["content_coverage"],
            "adjudication_calls": sum(len(f["adjudication"]["checks"]) for f in parsed["facts"]),
            "fidelity_pending": fidelity_pending,
            "seconds": time.monotonic() - started,
        }
        self.store.event({"event": "hybrid_extraction", **audit})
        return {"facts": accepted, "pending_facts": pending_facts, "rejected_facts": rejected,
                "pending_queries": queries, "uncovered": parsed["uncovered"],
                "entities": parsed["entities"], "sentences": sentences, "audit": audit}

    @staticmethod
    def focus(caption, item):
        a, b = item.get("span", [0, 0])
        left = caption[:a].split()[-3:]
        right = caption[b:].split()[:2]
        return " ".join(left + [caption[a:b]] + right).strip(" ,.;")
