"Image verification of facts (two independent templates)."
from __future__ import annotations

import hashlib
import json

from .adjudicate import _render

A_OPTIONS = {"supported": "The image clearly supports the claim.",
             "refuted": "The image clearly contradicts the claim.",
             "unclear": "The image does not show enough to decide."}


def template_b_settings(analysis, vcfg):
    """模板 B 的解码参数：template_b_max_new_tokens 为 null（或缺省）时用正常分析预算（规格 7.4）。
    pipeline.verify_fast 与 scripts/check_batching_equivalence.py 共用，避免两处口径不一致。"""
    cap = (vcfg or {}).get("template_b_max_new_tokens")
    return dict(analysis) if cap is None else dict(analysis, max_new_tokens=int(cap))


def claim_text(f):
    """Natural-language claim with explicit polarity/modality; JSON for legacy facts."""
    s = f.get("statement")
    if not s:
        return json.dumps({k: f[k] for k in ("subject", "predicate", "object", "scope", "polarity", "modality")},
                          ensure_ascii=False)
    body = s[:1].lower() + s[1:]
    if f["polarity"] == "negative" and f["modality"] == "uncertain":
        return "Possibly, it is not true that " + body
    if f["polarity"] == "negative":
        return "It is not true that " + body
    if f["modality"] == "uncertain":
        return "Possibly, " + body
    return s


PROTOCOL_A = ("Using only visible evidence in this image, assess the following claim. Its presence in this "
              "question is not evidence that it is true. Use supported only if the image clearly supports it, "
              "refuted only if the image clearly contradicts it, and unclear if the evidence is insufficient. "
              "Return JSON with status and evidence. Claim: %s")
PROTOCOL_B = ("Inspect the relevant entities and visual detail in the image first. Briefly state what is visible, "
              "then assess whether the following claim matches that observation. Do not infer hidden facts. "
              "Return JSON with observation, status (supported, refuted, or unclear), and evidence. Claim: %s")


def protocol_prompt(f, template):
    """The protocol's fixed template text with a natural-language claim."""
    return (PROTOCOL_A if template == 0 else PROTOCOL_B) % claim_text(f)


COCO_FOIL_OBJECTS = ["giraffe", "zebra", "elephant", "surfboard", "skis", "snowboard", "kite", "toaster",
                     "hair drier", "fire hydrant", "parking meter", "teddy bear", "toothbrush", "microwave",
                     "baseball glove", "tennis racket", "frisbee", "broccoli", "pizza", "donut", "sheep",
                     "cow", "bear", "airplane", "train", "boat", "umbrella", "suitcase", "laptop", "refrigerator"]
FOIL_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "brown"]


def _word_in(text, word):
    """Case-insensitive whole-word search (avoids 'train' matching 'train station' fragments)."""
    import re
    return re.search(r"(?<![a-z])" + re.escape(word.lower()) + r"(?![a-z])", text.lower()) is not None


def foil_facts(candidates, facts, n_objects=3, n_colors=2, seed=0):
    "Claims almost surely false for this image: objects named in NO caption, and colour"
    import random
    import re
    text = " ".join(c["text"].lower() for c in candidates)
    heads = {e.get("head", e["anchor"]).lower() for f in facts.values() for e in f.get("entities", [])}
    pool = [o for o in COCO_FOIL_OBJECTS if o not in text and o not in heads]
    rng = random.Random(seed)
    rng.shuffle(pool)
    foils, seen = [], set()
    for o in pool:
        statement = "There is %s %s." % ("an" if o[0] in "aeiou" else "a", o)
        if statement in seen:
            continue
        seen.add(statement)
        foils.append({"statement": statement, "polarity": "positive", "modality": "definite",
                      "foil": "absent_object", "value": o})
        if len([f for f in foils if f["foil"] == "absent_object"]) >= n_objects:
            break

    mentioned_colours = {c for c in FOIL_COLORS if _word_in(text, c)}
    colours = sorted((f for f in facts.values() if f["category"] == "attribute" and f.get("predicate") == "color"
                      and f["polarity"] == "positive" and f.get("entities")), key=lambda f: f.get("statement", ""))
    used = {}
    for f in facts.values():
        if f["category"] == "attribute" and f.get("predicate") == "color" and f.get("entities"):
            used.setdefault(f["entities"][0].get("key", f["entities"][0]["anchor"]), set()).add(f["object"].lower())
    added = 0
    for f in colours:
        if added >= n_colors:
            break
        key = f["entities"][0].get("key", f["entities"][0]["anchor"])
        original = (f.get("object") or "").lower()
        alt = [c for c in FOIL_COLORS if c not in used.get(key, set()) and c not in original
               and c not in mentioned_colours]
        if not alt:
            continue
        colour = rng.choice(alt)
        statement = f.get("statement") or ""
        statement_source = "templated"
        if original and _word_in(statement, original):

            statement = re.sub(r"(?<![a-z])" + re.escape(original) + r"(?![a-z])", colour,
                               statement, count=1, flags=re.I)
            statement_source = "substituted"
        else:
            name = f["entities"][0].get("display", f["entities"][0]["anchor"])
            statement = "The %s is %s." % (name, colour)
        if statement in seen:
            continue
        seen.add(statement)
        foils.append({"statement": statement, "polarity": "positive", "modality": "definite",
                      "foil": "swapped_color", "source_fact": f.get("id"), "value": colour,
                      "original_color": original or None, "statement_source": statement_source})
        added += 1
    return foils

def soft_values(raw, mapping):
    p = {mapping[k]: v for k, v in raw["probs"].items()}
    return {"supported": p.get("supported", 0.0), "refuted": p.get("refuted", 0.0), "unclear": p.get("unclear", 0.0),
            "option_mass": raw.get("mass"), "mapping": mapping}


def prompt_a(f):
    q = ("Look at the image and judge the claim using only what is visible. The claim may be wrong; "
         "its presence here is not evidence that it is true.\nClaim: \"%s\"" % claim_text(f))
    return _render(q, A_OPTIONS, "verifyA|" + claim_text(f), fixed_last="unclear")


def prompt_b(f):
    return ("Inspect the relevant objects in the image first, then judge the claim using only visible "
            "evidence. Do not infer hidden facts. The claim may be wrong.\nClaim: \"%s\"\n"
            "Return JSON only, with keys in this order: observation (one short sentence about what is "
            "visible), status (supported, refuted, or unclear), evidence (a few words)." % claim_text(f))


def parse_a(raw, mapping):
    best = max(raw["probs"], key=raw["probs"].get)
    status = mapping[best]
    p = {mapping[k]: v for k, v in raw["probs"].items()}
    return {"status": status, "evidence": "option probabilities %s" % json.dumps({k: round(v, 4) for k, v in p.items()}),
            "probabilities": p, "option_mass": raw.get("mass"), "mapping": mapping,
            "prompt_sha": hashlib.sha256(json.dumps(mapping).encode()).hexdigest()[:8]}


def in_scope(f, scope):
    """'scored': only facts that any score uses (definite claims: coverage, q, error
    constraint). 'all': every fact (legacy protocol)."""
    return scope == "all" or f["modality"] == "definite"


YESNO_SUFFIX = "Answer with exactly Yes or No. Do not use True/False, other words, or explanations."
YESNO_WORDS = ["No", "Yes"]
YESNO_A = 'Is the following statement about the image true?\nStatement: "%s"\n' + YESNO_SUFFIX
YESNO_B = "Look closely at %s in the image. Based only on what is visible, is it true that %s?\n" + YESNO_SUFFIX


def base_statement(f):
    """事实的肯定、确定基本句（7.2）；没有 statement 说明 B5 没修好，直接报错。"""
    s = f.get("statement")
    if not s:
        raise ValueError("yes/no verification needs fact['statement']")
    return s


def yesno_targets(f):
    names = []
    for e in (f.get("entities") or [])[:3]:
        name = "the " + (e.get("display") or e["anchor"])
        if name not in names:
            names.append(name)
    return " and ".join(names) if names else "the whole image"


def yesno_prompt(f, template):
    s = base_statement(f)
    if template == 0:
        return YESNO_A % s
    body = s[:1].lower() + s[1:]
    body = body[:-1] if body.endswith(".") else body
    return YESNO_B % (yesno_targets(f), body)


def yesno_check(raw, f, band=(0.35, 0.65), min_mass=0.3):
    """raw = backend.choice / choice_batch 对 YESNO_WORDS 的返回值；按 7.4 判定，返回 7.5 的字段。"""
    lo, hi = float(band[0]), float(band[1])
    p_yes_raw, p_no_raw = raw["raw_probs"]["Yes"], raw["raw_probs"]["No"]
    mass = p_yes_raw + p_no_raw
    p_yes = p_yes_raw / mass if mass > 0 else 0.5
    low_mass = mass < min_mass
    if low_mass:
        base = "unclear"
    elif p_yes >= hi:
        base = "supported"
    elif p_yes <= lo:
        base = "refuted"
    else:
        base = "unclear"
    negative = f["polarity"] == "negative"
    status = {"supported": "refuted", "refuted": "supported"}.get(base, base) if negative else base
    p_true = 1 - p_yes if negative else p_yes
    return {"artifact_type": "verification", "status": status, "base_status": base,
            "p_yes": p_yes, "p_true": p_true, "option_mass": mass, "low_mass": low_mass, "band": [lo, hi],
            "evidence": "P(yes)=%.4f for the affirmative statement; polarity %s" % (p_yes, f["polarity"])}
