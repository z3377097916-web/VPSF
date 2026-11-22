"Stage B of hybrid extraction: closed-choice adjudication by the frozen LLaVA."
from __future__ import annotations

import hashlib
import re

CLAIM_OPTIONS = {
    "definite": "The text states it as true.",
    "possible": "The text says it is only possibly true (for example with may, might, appears, likely, possibly, or as one of several alternatives).",
    "negated": "The text states that it is not true.",
    "absent": "The text does not say it.",
}
EXPECTED = {
    ("positive", "definite"): {"definite"},
    ("positive", "uncertain"): {"possible"},
    ("negative", "definite"): {"negated"},
    ("negative", "uncertain"): {"possible", "negated"},
}
LETTERS = "ABCDEFG"
INSTRUCTION = "Answer with the option's letter from the given choices directly."
NONE = "none of these"


INFO_RISKS = {"clause_cue_not_applied", "presupposed_location", "place_as_setting", "cue", "alternative"}
ROLE_CATEGORIES = {"attribute", "spatial", "relation", "action", "identity", "other", "text_ocr"}


def _perm(keys, salt):
    h = hashlib.sha256(salt.encode("utf-8")).digest()
    return [keys[i] for i in sorted(range(len(keys)), key=lambda i: (h[i % len(h)], i))]


def _render(question, options, salt, fixed_last=None):
    keys = list(options)
    movable = [k for k in keys if k != fixed_last]
    order = _perm(movable, salt) + ([fixed_last] if fixed_last in keys else [])
    lines = [question] + ["%s. %s" % (LETTERS[i], options[k]) for i, k in enumerate(order)] + [INSTRUCTION]
    return "\n".join(lines), {LETTERS[i]: k for i, k in enumerate(order)}


INVERSE = {
    "on top of": "under", "under": "above", "above": "below", "below": "above",
    "over": "below", "underneath": "above", "beneath": "above",
    "in front of": "behind", "behind": "in front of",
    "to the left of": "to the right of", "to the right of": "to the left of",
    "on the left side of": "on the right side of", "on the right side of": "on the left side of",
    "towards the left side": "towards the right side", "towards the right side": "towards the left side",
    "left of": "right of", "right of": "left of",
    "inside": "contains", "contains": "inside", "contain": "inside",
}
SYMMETRIC_PREDICATES = ("next to", "beside", "near", "between", "with", "along", "around")


def inverse_predicate(pred):
    """Return a safe converse consequence of an exact predicate; otherwise None."""
    p = (pred or "").lower().strip()

    return INVERSE.get(p)


def _entity_label(entity):
    name = entity.get("display") or entity.get("anchor") or "entity"
    return name if name.lower().startswith("the ") else "the " + name


def _direction_role(fact, swap):
    ents = fact.get("entities") or []
    if len(ents) < 2 or fact.get("polarity") == "negative":
        return None
    first, second = ents[:2]
    pred = fact.get("predicate") or ""
    if swap:
        inv = inverse_predicate(pred)
        if not inv:
            return None
        subject, other, pred = second, first, inv
    else:
        subject, other = first, second
    subj = _entity_label(subject)
    obj = _entity_label(other)

    if not inverse_predicate(fact.get("predicate")) and fact.get("category") != "spatial":
        return None
    phrase = "contains" if pred in {"contain", "contains"} else "is " + pred
    return {"question": "which of these %s %s?" % (phrase, obj),
            "answer": subj, "sentence_entities": [_entity_label(e) for e in ents], "co_subjects": []}


def direction_applicable(fact):
    return fact.get("category") in {"spatial", "relation"} and len(fact.get("entities") or []) >= 2


def _slot_check(fact, sentences, entity, kind):
    """Keep the complete proposition (including recipients), mask one named role."""
    name = entity.get("display") or entity["anchor"]
    statement = fact.get("statement", "")
    pattern = re.compile(r"(?<!\w)(?:(?:the|a|an)\s+)?" + re.escape(name) + r"(?!\w)", re.I)
    if len(list(pattern.finditer(statement))) != 1:
        return None
    masked = pattern.sub("[TARGET]", statement, count=1)
    if fact.get("polarity") == "negative":
        masked = "It is not true that " + masked[:1].lower() + masked[1:]
    if fact.get("modality") == "uncertain":
        masked = "Possibly, " + masked
    q = ('Read the text.\nText: "%s"\nAccording to the text, which entity fills [TARGET] '
         'in the following statement?\nStatement: "%s"' % (context_text(fact, sentences), masked))
    expected = _entity_label(entity)
    names = list(dict.fromkeys([expected] + [_entity_label(e) for e in fact["entities"]]))[:4]
    prompt, mapping = _render(q, {**{x: x for x in names}, NONE: NONE}, kind + q, fixed_last=NONE)
    return kind, prompt, mapping, expected


def direction_checks(fact, sentences):
    "Closed questions that verify roles in a binary or multi-participant relation."
    if not direction_applicable(fact):
        return []
    out = []
    if not fact.get("role"):
        role = _direction_role(fact, False)
        if role:
            synth = dict(fact, role=role)
            built = role_question(synth, sentences)
            if built:
                out.append(("direction_forward", built[0], built[1], role["answer"]))
        else:
            check = _slot_check(fact, sentences, fact["entities"][0], "direction_forward")
            if check:
                out.append(check)
    role = _direction_role(fact, True)
    if role:
        synth = dict(fact, role=role)
        built = role_question(synth, sentences)
        if built:
            out.append(("direction_reverse", built[0], built[1], role["answer"]))


    for i, entity in enumerate(fact["entities"][1:], 1):
        if i == 1 and role is not None:
            continue
        if i == 1 and len(fact["entities"]) == 2 and fact.get("predicate", "").lower() in SYMMETRIC_PREDICATES:
            continue
        check = _slot_check(fact, sentences, entity, "direction_role_%d" % i)
        if check:
            out.append(check)
    return out


def context_text(fact, sentences):
    k = fact["sent_index"]
    if fact.get("coref"):
        indexes = {k}
        for link in fact["coref"]:
            indexes.update(e["sent_index"] for e in link.get("evidence", []) if "sent_index" in e)
        if len(indexes) == 1:
            indexes.add(max(0, k - 1))
        return " ".join(sentences[i] for i in sorted(indexes) if 0 <= i < len(sentences))
    return sentences[k]


def role_question(fact, sentences, max_options=4, exclude_named=True):
    role = fact.get("role")
    if not role:
        return None
    text = context_text(fact, sentences)
    names = [n for n in role["sentence_entities"] if n not in role.get("co_subjects", [])]
    if exclude_named:


        q = role["question"].lower()
        names = [n for n in names if n == role["answer"] or n[4:].lower() not in q]
    if role["answer"] not in names:
        names.insert(0, role["answer"])
    names = [role["answer"]] + [n for n in names if n != role["answer"]][:max_options - 1]
    options = {n: n for n in names}
    options[NONE] = NONE
    q = 'Read the text.\nText: "%s"\nAccording to the text, %s' % (text, role["question"])
    return _render(q, options, "role|" + text + "|" + role["question"], fixed_last=NONE)


def claim_question(fact, sentences):
    text = context_text(fact, sentences)
    q = ('Read the text and judge only what the text says, not what is likely in reality.\n'
         'Text: "%s"\nStatement: "%s"\nHow does the text present the statement?' % (text, fact["statement"]))
    return _render(q, CLAIM_OPTIONS, "claim|" + text + "|" + fact["statement"], fixed_last="absent")


def coref_question(coref, fact, sentences):
    options = {"antecedent": "the " + coref["antecedent"]}
    for i, alt in enumerate(coref.get("alternatives", [])[:3]):
        options["alt%d" % i] = "the " + alt
    options["unclear"] = "It cannot be determined from the text."
    text = context_text(fact, sentences)
    q = ('Read the text.\nText: "%s"\nIn the sentence "%s", what does the word "%s" refer to?'
         % (text, sentences[fact["sent_index"]], coref["pronoun"]))
    return _render(q, options, "coref|%s|%s|%s" % (text, coref["pronoun"], coref["pronoun_span"]), fixed_last="unclear")


class Adjudicator:
    """Ask the frozen model closed questions; return verdicts without editing facts."""

    def __init__(self, backend, store, mode="risky", reject_threshold=0.6, min_mass=0.3,
                 calibrate=False, claim_check=False, seed=42, direction_consistency=False):
        if mode not in {"risky", "all", "none"}:
            raise ValueError("adjudicate mode must be risky, all or none")
        self.backend, self.store, self.mode = backend, store, mode
        self.reject_threshold, self.min_mass = reject_threshold, min_mass
        self.calibrate, self.claim_check, self.seed = calibrate, claim_check, seed
        self.direction_consistency = direction_consistency
        self._baseline = {}

    def ask(self, kind, prompt, mapping):
        letters = sorted(mapping)
        request = {"kind": "adjudicate_" + kind, "prompt": prompt, "letters": letters,
                   "history": [], "seed": self.seed}
        raw = self.store.cached(request, lambda: self.backend.choice(prompt, letters))
        if raw.get("history_messages", 0):
            raise ValueError("HISTORY_REVIEW: adjudication must be a fresh request")
        probs = dict(raw["probs"])
        if self.calibrate:
            base = self.baseline(len(letters))
            probs = {k: probs[k] / max(base[k], 1e-6) for k in probs}
            z = sum(probs.values()) or 1.0
            probs = {k: v / z for k, v in probs.items()}
        best = max(probs, key=probs.get)
        return {"kind": kind, "prompt": prompt, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                "mapping": mapping, "probs": probs, "mass": raw.get("mass"),
                "answer": mapping[best], "p": probs[best]}

    def prefetch(self, facts, sentences, batch_size=16):
        """Answer all role / coref questions of one caption in batched forward passes.
        Results go into the same cache judge() reads, so decisions are unchanged."""
        if not hasattr(self.backend, "choice_batch") or not hasattr(self.store, "peek"):
            return
        pending = {}
        for f in facts:
            if not self.needs_check(f):
                continue
            built = []
            if self.claim_check and (self.mode == "all" or f.get("cues")):
                built.append(("claim",) + claim_question(f, sentences))
            if self.role_needed(f):
                built.append(("role",) + role_question(f, sentences))
            for coref in f.get("coref", []):
                if coref.get("requires_check") or coref.get("alternatives") or self.mode == "all":
                    built.append(("coref",) + coref_question(coref, f, sentences))
            if self.direction_consistency:
                for kind, prompt, mapping, _ in direction_checks(f, sentences):
                    built.append((kind, prompt, mapping))
            for kind, prompt, mapping in built:
                letters = sorted(mapping)
                req = {"kind": "adjudicate_" + kind, "prompt": prompt, "letters": letters, "history": [], "seed": self.seed}
                if self.store.peek(req) is None:
                    pending.setdefault(tuple(letters), {})[prompt] = req
        for letters, reqs in pending.items():
            items = list(reqs.values())
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                for req, res in zip(chunk, self.backend.choice_batch([r["prompt"] for r in chunk], list(letters))):
                    self.store.put(req, res)

    def baseline(self, n):
        """Content-free prior over option letters (contextual calibration; optional)."""
        if n not in self._baseline:
            letters = list(LETTERS[:n])
            prompt = 'Read the text.\nText: "N/A"\nAccording to the text, what is N/A?\n' + \
                "\n".join("%s. N/A" % x for x in letters) + "\n" + INSTRUCTION
            request = {"kind": "adjudicate_baseline", "prompt": prompt, "letters": letters, "history": []}
            self._baseline[n] = self.store.cached(request, lambda: self.backend.choice(prompt, letters))["probs"]
        return self._baseline[n]

    def role_needed(self, fact):
        if fact["category"] not in ROLE_CATEGORIES or not fact.get("role"):
            return False
        if self.mode == "all":
            return True
        role = fact["role"]
        others = [n for n in role["sentence_entities"] if n != role["answer"] and n not in role.get("co_subjects", [])]
        return bool(others) or bool(set(fact.get("risk", [])) - INFO_RISKS)

    def needs_check(self, fact):
        if self.mode == "none":
            return False
        if self.direction_consistency and direction_applicable(fact):
            return True
        if self.mode == "all":
            return bool(fact.get("role")) or bool(fact.get("coref")) or self.claim_check
        return self.role_needed(fact) or bool(fact.get("coref")) or (self.claim_check and bool(fact.get("cues")))

    def low_mass(self, answer):
        return answer["mass"] is not None and answer["mass"] < self.min_mass

    def judge(self, fact, sentences):
        """Return (status, record). status in {'accepted', 'pending', 'rejected'}."""
        record = {"checks": [], "mode": self.mode}
        if not self.needs_check(fact):
            record["reason"] = "no_ambiguity_parser_fact"
            return "accepted", record
        if self.claim_check and (self.mode == "all" or fact.get("cues")):
            claim = self.ask("claim", *claim_question(fact, sentences))
            record["checks"].append(claim)
            if self.low_mass(claim):
                record["reason"] = "low_option_mass"
                return "pending", record
            if claim["answer"] not in EXPECTED[(fact["polarity"], fact["modality"])]:
                if claim["answer"] == "absent" and claim["p"] >= self.reject_threshold:
                    record["reason"] = "text_does_not_state_candidate"
                    return "rejected", record
                record["reason"] = "claim_disagreement:" + claim["answer"]
                return "pending", record
        if self.role_needed(fact):
            built = role_question(fact, sentences)
            r = self.ask("role", *built)
            record["checks"].append(r)
            if self.low_mass(r):
                record["reason"] = "low_option_mass"
                return "pending", record
            if r["answer"] == NONE and r["p"] >= self.reject_threshold:
                record["reason"] = "role:none"
                return "rejected", record
            if r["answer"] != fact["role"]["answer"]:
                record["reason"] = "role:" + r["answer"]
                return "pending", record
        for index, coref in enumerate(fact.get("coref", [])):
            if not coref.get("requires_check") and not coref.get("alternatives") and self.mode != "all":
                continue
            c = self.ask("coref", *coref_question(coref, fact, sentences))
            c["coref_index"] = index
            record["checks"].append(c)
            if self.low_mass(c) or c["answer"] != "antecedent":
                record["reason"] = "coref:" + c["answer"]
                return "pending", record
            c["validated"] = True
        if self.direction_consistency:
            for kind, prompt, mapping, expected in direction_checks(fact, sentences):

                r = self.ask(kind, prompt, mapping)
                record["checks"].append(r)
                if self.low_mass(r):
                    record["reason"] = "low_option_mass"
                    return "pending", record
                if r["answer"] != expected:
                    record["reason"] = "direction_conflict:" + r["answer"]
                    return "pending", record
        record["reason"] = "agree"
        return "accepted", record
