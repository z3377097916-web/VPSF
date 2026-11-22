"Deterministic caption -> candidate atomic facts (stage A of hybrid extraction)."
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field

from . import lexicon as L
from .semantics import semantic_payload

PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "himself", "herself",
            "themselves", "itself"}
POSSESSIVE_PRON = {"his", "its", "their", "her", "hers", "theirs"}
REL_PRONOUNS = {"who", "that", "which", "whom", "whose"}
NOUN_POS = {"NOUN", "PROPN"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJ_DEPS = {"dobj", "obj"}
COPULA = {"be"}
OCR_PATTERN = re.compile(r'"([^"]{1,80})"|“([^”]{1,80})”|(?<=\s)\'([^\']{1,40})\'(?=[\s.,;:!?])')
BW_PATTERN = re.compile(r'\bblack[- ]and[- ]white\b', re.I)
PLACE_PREPS = {"in", "at", "inside", "through", "on", "across", "along", "near", "around", "into"}


def low(t):
    return t.lower_


def lem(t):
    return (t.lemma_ or t.text).lower()


def char_span(tokens):
    tokens = sorted(tokens, key=lambda t: t.i)
    return [tokens[0].idx, tokens[-1].idx + len(tokens[-1].text)]


def is_plural(t):
    return t.tag_ in {"NNS", "NNPS"} or low(t) in {
        "people", "men", "women", "children", "police", "crowd", "family", "couple",
        "group", "team", "audience", "staff", "cattle", "sheep", "deer", "fish"}


def number_value(text):
    """Normalise a count phrase: 'Two' -> '2', 'at least three' -> 'at least 3'."""
    t = text.lower().replace("-", " ").strip()
    qualifier = ""
    for q in sorted(L.COUNT_QUALIFIERS, key=len, reverse=True):
        if t.startswith(q + " "):
            qualifier, t = q, t[len(q):].strip()
            break
    core = " ".join(w for w in t.split() if w not in {"a", "an", "the", "of"})
    if core.isdigit():
        n = core
    elif core in L.NUMBER_WORDS:
        n = str(L.NUMBER_WORDS[core])
    else:
        n = core
    return (qualifier + " " + n).strip()


def inflect_3sg(verb):
    v = verb.lower()
    if v == "be":
        return "is"
    if v == "have":
        return "has"
    if re.search(r"(s|sh|ch|x|z|o)$", v):
        return v + "es"
    if re.search(r"[^aeiou]y$", v):
        return v[:-1] + "ies"
    return v + "s"


@dataclass
class Entity:
    id: str
    head_key: str
    anchor: str
    qualifiers: list
    plural: bool
    person: bool
    gender: str
    first_sent: int
    mentions: list = field(default_factory=list)
    distinguish: bool = False

    def key(self):
        if self.distinguish and self.qualifiers:
            return self.head_key + "[" + ",".join(sorted(q.lower() for q in self.qualifiers)) + "]"
        return self.head_key

    @property
    def display(self):
        if self.distinguish and self.qualifiers:
            return " ".join(self.qualifiers) + " " + self.anchor
        return self.anchor

    def as_dict(self):

        return {"id": self.id, "anchor": self.anchor, "qualifiers": list(self.qualifiers),
                "head": self.head_key, "key": self.key(), "display": self.display, "plural": self.plural}


@dataclass
class Ref:
    """A participant slot: one entity, or an or-alternative group ("a cat or a dog")."""
    entities: list
    text: str
    tokens: list
    alternative: bool = False
    coref: dict | None = None

    def key(self):
        return "|".join(sorted(e.key() for e in self.entities))

    @property
    def plural(self):
        if self.alternative:
            return self.entities[0].plural
        return len(self.entities) > 1 or self.entities[0].plural


class CaptionParser:
    """Rule-based dependency reading of one caption (spaCy, ClearNLP labels)."""

    def __init__(self, nlp, caption):
        self.caption = caption
        self.doc = nlp(caption)
        self._merge_hyphenated()
        self.sents = list(self.doc.sents)
        self.sent_of = {t.i: k for k, s in enumerate(self.sents) for t in s}
        self.entities: list[Entity] = []
        self.mention_entity = {}
        self.facts = []
        self.pending = []
        self.used = set()
        self.ocr_tokens, self.style_tokens = set(), set()
        self.consumed_preps = set()
        for pattern, target in ((OCR_PATTERN, self.ocr_tokens), (BW_PATTERN, self.style_tokens)):
            for m in pattern.finditer(caption):
                a, b = m.span()
                target.update(t.i for t in self.doc if a <= t.idx < b)

    def _merge_hyphenated(self):
        """'double-decker', 'black-and-white', 'top-left' -> one token (spaCy only)."""
        if not hasattr(self.doc, "retokenize"):
            return
        spans = []
        for m in re.finditer(r"\b\w+(?:-\w+)+\b", self.caption):
            span = self.doc.char_span(m.start(), m.end())
            if span is not None and len(span) > 1:
                spans.append(span)
        if spans:
            with self.doc.retokenize() as r:
                for span in spans:
                    root = span.root
                    r.merge(span, attrs={"LEMMA": span.text.lower(), "POS": root.pos_, "TAG": root.tag_})


    def surface(self, tokens):
        a, b = char_span(tokens)
        return self.caption[a:b]

    def np_value_tokens(self, t):
        """Noun phrase without determiners/possessives: 'a large grassy field' -> 'large grassy field'."""
        toks = [t]
        for c in t.lefts:
            if c.dep_ in {"amod", "compound", "nummod", "quantmod", "nmod"} and c.i not in self.ocr_tokens:
                toks.extend(c.subtree)
        return sorted(set(toks), key=lambda x: x.i)

    def group_tokens(self, items):
        toks = []
        for x in items:
            toks.extend(self.np_value_tokens(x))
        a, b = min(x.i for x in toks), max(x.i for x in toks)
        return [self.doc[i] for i in range(a, b + 1)]


    def of_object(self, t):
        for c in t.children:
            if c.dep_ == "prep" and low(c) == "of":
                for g in c.children:
                    if g.dep_ == "pobj":
                        return g
        return None

    def is_predicate_nominal(self, t):
        """'a museum' in 'The building may be a museum' (identity, not an entity)."""
        x = t
        while x.dep_ == "conj":
            x = x.head
        h = x.head
        return x.dep_ == "attr" and lem(h) in COPULA and not any(c.dep_ == "expl" for c in h.children)

    def is_image_reference(self, t):
        x = t
        if x.dep_ == "attr" and any(lem(s) in L.META_NOUNS for s in x.head.children if s.dep_ in SUBJ_DEPS):
            return True
        if self.of_object(x) is not None and x.dep_ in {"nsubj", "ROOT", "attr", "dobj"}:
            return True
        return False

    @staticmethod
    def is_gerund_clause(t):
        """'with a red bus driving past buildings': spaCy may tag 'driving' NOUN with an nsubj."""
        return t.pos_ in NOUN_POS and low(t).endswith("ing") and any(c.dep_ in SUBJ_DEPS for c in t.children)

    def is_entity_head(self, t):
        if t.i in self.ocr_tokens or t.i in self.style_tokens or t.pos_ not in NOUN_POS:
            return False
        if self.is_gerund_clause(t):
            return False
        if t.dep_ == "compound" and t.head.pos_ in NOUN_POS:
            return False
        w = lem(t)
        if w in L.META_NOUNS or w in L.ABSTRACT or w in L.TIME or low(t) in L.PARTITIVE:
            return False
        if w in L.POSITION_NOUNS and (t.dep_ == "pobj" or any(c.dep_ == "prep" for c in t.children)):
            return False
        if w in (L.QUANT_NOUNS | L.PART_NOUNS) and self.of_object(t) is not None:
            return False
        if w in L.STYLE_NOUNS and self.is_image_reference(t):
            return False
        if self.is_predicate_nominal(t):
            return False
        return True

    def qualifiers(self, t):
        out = []
        for c in t.children:
            if c.i in self.ocr_tokens or c.i in self.style_tokens:
                continue
            if self.lexicalized(c, t):
                continue
            if c.dep_ == "amod" and low(c) not in L.HEDGE_ADV and low(c) not in L.SUBJECTIVE:
                if low(c) in L.NON_ATTRIBUTE_ADJ and low(c) not in L.DISTINCT_MARKERS:
                    continue
                out.append(self.modifier_text(c))
            elif c.dep_ == "compound" and lem(c) in L.MATERIALS:
                out.append(c.text)
            elif c.dep_ == "det" and low(c) in L.DISTINCT_MARKERS:
                out.append(low(c))
        return out

    def modifier_text(self, c):
        toks = [c] + [g for g in c.children if g.dep_ in {"advmod", "amod", "npadvmod", "compound"}
                      and low(g) in (L.SHADES | {"very"})]
        return self.surface(toks)

    def lexicalized(self, c, t):
        return c.dep_ == "amod" and (low(c) + " " + low(t)) in L.LEXICALIZED

    def anchor_tokens(self, t):
        toks = [t]
        for c in t.lefts:
            if self.lexicalized(c, t):
                toks.append(c)
                continue
            if c.dep_ == "compound" and lem(c) not in L.MATERIALS and c.i not in self.ocr_tokens:
                toks.append(c)
                toks.extend(g for g in c.lefts if g.dep_ == "compound")
        toks = sorted(set(toks), key=lambda x: x.i)
        while len(toks) > 1 and toks[-1].i - toks[0].i + 1 != len(toks):
            toks = toks[1:]
        return toks

    def entity_for(self, t):
        if t.i in self.mention_entity:
            return self.mention_entity[t.i]
        toks = self.anchor_tokens(t)
        key = " ".join(lem(x) for x in toks)
        quals = self.qualifiers(t)
        distinct = any(q in L.DISTINCT_MARKERS for q in quals)
        same = [e for e in self.entities if e.head_key == key]
        if same and t.tag_ in {"NN", "NNP"} and all(e.plural for e in same) and any(
                (c.dep_ == "nummod" and low(c) == "one") or (c.dep_ == "det" and low(c) in {"one", "another", "each"})
                for c in t.children):
            distinct = True
            quals = quals + [next(low(c) for c in t.children if low(c) in {"one", "another", "each"})]
        chosen = None
        if same and not distinct:
            for e in same:
                if not self.qualifier_conflict(e.qualifiers, quals):
                    chosen = e
                    e.qualifiers.extend(q for q in quals if q not in e.qualifiers)
                    break
        if chosen is None:
            chosen = Entity(id="e%d" % (len(self.entities) + 1), head_key=key, anchor=self.surface(toks),
                            qualifiers=list(quals), plural=is_plural(t),
                            person=lem(t) in L.PERSON_NOUNS or low(t) in L.PERSON_NOUNS or t.ent_type_ == "PERSON",
                            gender="m" if low(t) in L.MALE else "f" if low(t) in L.FEMALE else "",
                            first_sent=self.sent_of[t.i])
            if same:
                chosen.distinguish = True
                for e in same:
                    e.distinguish = True
            self.entities.append(chosen)
        chosen.mentions.append(t.i)
        self.mention_entity[t.i] = chosen
        return chosen

    @staticmethod
    def qualifier_conflict(a, b):
        def classes(qs):
            out = {}
            for q in qs:
                w = q.lower().split()[-1]
                for name, vocab in (("color", L.COLORS), ("material", L.MATERIALS),
                                    ("size", L.SIZES), ("pattern", L.PATTERNS), ("position", L.POSITION_ADJ)):
                    if w in vocab:
                        out.setdefault(name, set()).add(q.lower())
            return out
        ca, cb = classes(a), classes(b)
        return any(k in cb and ca[k].isdisjoint(cb[k]) for k in ca)


    def group_head(self, t):
        x = t
        while x.dep_ == "conj" and x.head.pos_ == t.pos_:
            x = x.head
        return x

    def conjuncts(self, t):
        items, disj, stack = [t], False, [t]
        while stack:
            x = stack.pop()
            for c in x.children:
                if c.dep_ == "cc" and low(c) in {"or", "nor"}:
                    disj = True
                if c.dep_ == "conj" and c.pos_ == t.pos_:
                    items.append(c)
                    stack.append(c)
        return sorted(items, key=lambda x: x.i), disj

    def single(self, t):
        e = self.entity_for(t)
        return Ref([e], e.anchor, self.anchor_tokens(t))

    def ref(self, t):
        """Participant Refs for token t: pronoun resolution, 'and' expansion, 'or' grouping."""
        if t is None:
            return []
        if low(t) in REL_PRONOUNS and t.pos_ in {"PRON", "DET"}:
            v = t.head


            while v.dep_ not in {"relcl", "acl"} and v.head.i != v.i:
                v = v.head
            return self.ref(v.head) if v.dep_ in {"relcl", "acl"} else []
        if low(t) in PRONOUNS:
            return self.resolve_pronoun(t)
        if t.pos_ == "PRON" or low(t) in L.PARTITIVE:

            self.pending.append(self.pending_item(t, "subgroup or partitive reference '%s'" % t.text, []))
            return []
        if lem(t) in (L.QUANT_NOUNS | L.PART_NOUNS) and self.of_object(t) is not None:
            return self.ref(self.of_object(t))
        if not self.is_entity_head(t):
            return []
        head = self.group_head(t)
        items, disj = self.conjuncts(head)
        items = [x for x in items if self.is_entity_head(x)]
        if disj and len(items) > 1:
            ents = [self.entity_for(x) for x in items]
            toks = self.group_tokens(items)
            det = [c for c in items[0].lefts if c.dep_ == "det" and c.i < toks[0].i]
            text = self.surface(det + toks)
            return [Ref(ents, text, toks, alternative=True)]
        if t is not head:
            return [self.single(t)]
        return [self.single(x) for x in items]

    def resolve_pronoun(self, t):
        w = low(t)
        k = self.sent_of[t.i]
        candidates = []
        for e in self.entities:
            earlier = [m for m in e.mentions if m < t.i]
            if not earlier:
                continue
            if w in {"they", "them", "themselves"}:
                ok = e.plural
            elif w in {"he", "him", "himself"}:
                ok = e.person and e.gender in {"", "m"} and not e.plural
            elif w in {"she", "her", "herself"}:
                ok = e.person and e.gender in {"", "f"} and not e.plural
            else:
                ok = not e.person and not e.plural
            if ok:
                candidates.append((max(earlier), e))
        if not candidates:
            self.pending.append(self.pending_item(t, "unresolved pronoun '%s'" % t.text, []))
            return []

        def salience(item):
            last, _ = item
            return (self.sent_of[last] >= k - 1, self.doc[last].dep_ in SUBJ_DEPS, last)
        ranked = sorted(candidates, key=salience, reverse=True)
        best = ranked[0][1]
        alternatives = [e for last, e in ranked[1:] if self.sent_of[last] >= k - 1]
        coref = {"pronoun": t.text, "pronoun_span": char_span([t]), "antecedent": best.anchor,
                 "antecedent_id": best.id, "alternatives": [e.anchor for e in alternatives],
                 "alternative_ids": [e.id for e in alternatives]}
        return [Ref([best], best.anchor, [t], coref=coref)]


    def governing_verb(self, t):
        x, n = t, 0
        while x.pos_ not in {"VERB", "AUX"} and x.head.i != x.i and n < 30:
            x, n = x.head, n + 1
        return x if x.pos_ in {"VERB", "AUX"} else None

    def clause_cues(self, v, depth=0):
        """(negated, uncertain, cue_tokens) for the clause headed by v."""
        if v is None or depth > 8:
            return False, False, []
        neg, unc, cues = False, False, []
        for c in v.children:
            if c.dep_ == "neg" or (c.dep_ == "advmod" and low(c) == "never"):
                neg = True
                cues.append(c)
            elif c.dep_ in {"aux", "auxpass"} and lem(c) in L.MODALS:
                unc = True
                cues.append(c)
            elif c.dep_ in {"advmod", "npadvmod"} and low(c) in L.HEDGE_ADV:
                unc = True
                cues.append(c)
        h = v.head
        if h is not v:
            if v.dep_ in {"xcomp", "ccomp"} and lem(h) in L.HEDGE_RAISING:
                unc = True
                cues.append(h)
            if v.dep_ in {"ccomp", "advcl", "xcomp", "acl"} and lem(h) in L.HEDGE_INFER:
                unc = True
                cues.append(h)
            if v.dep_ in {"xcomp", "advcl"} and lem(h) in L.INTENT_VERBS and any(c.dep_ == "aux" and low(c) == "to" for c in v.children):
                unc = True
                cues.append(h)
            if v.dep_ in {"xcomp", "ccomp"}:
                _, u2, c2 = self.clause_cues(h, depth + 1)
                if u2:
                    unc = True
                    cues.extend(c2)
            if v.dep_ == "conj" and not any(c.dep_ in SUBJ_DEPS for c in v.children):
                n2, u2, c2 = self.clause_cues(h, depth + 1)
                neg, unc = neg or n2, unc or u2
                cues.extend(c2)
        return neg, unc, cues

    def noun_cues(self, t):
        neg, unc, cues = False, False, []
        x = self.group_head(t)
        for node in {t, x}:
            for c in node.children:
                if c.dep_ == "det" and low(c) in {"no", "neither"}:
                    neg = True
                    cues.append(c)
                elif c.dep_ in {"advmod", "amod"} and low(c) in L.HEDGE_ADV:
                    unc = True
                    cues.append(c)
        return neg, unc, cues


    def focus_phrase(self, t):
        """Minimal clue: the word plus its directly linked word, never a wider text window."""
        toks = [t]
        h = t.head
        if h is not t and abs(h.i - t.i) <= 3 and h.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"} \
                and h.i not in self.ocr_tokens:
            toks.append(h)
        else:
            obj = next((c for c in t.children if c.dep_ in OBJ_DEPS | {"pobj"} and abs(c.i - t.i) <= 3), None)
            if obj is not None:
                toks.append(obj)
        words = []
        for x in sorted(toks, key=lambda x: x.i):
            if x.lower_ not in [w.lower() for w in words]:
                words.append(x.text)
        return " ".join(words)

    def pending_item(self, t, issue, anchors):
        k = self.sent_of[t.i]
        return {"quote": self.sents[k].text, "issue": issue, "anchors": anchors,
                "span": char_span([t]), "sent_index": k, "focus": self.focus_phrase(t)}

    def emit(self, category, subj, predicate, pred_key, obj=None, value="", value_key=None,
             polarity="positive", modality="definite", rule="", tokens=(), cues=(), scope="",
             subtype="", extra=(), risk=(), statement=None, sent=0, reverse=None):
        risk = list(risk)
        if subj is not None and subj.alternative:
            modality = "uncertain"
        entities = []
        for r in (subj, obj, *extra):
            if r is None:
                continue
            entities.extend(r.entities)
            if r.coref:
                risk.append("coref")
            if r.alternative:
                risk.append("alternative")
        if cues:
            risk.append("cue")
        toks = list(tokens) + list(cues)
        for r in (subj, obj, *extra):
            if r is not None:
                toks.extend(r.tokens)
        self.used.update(t.i for t in toks)
        uniq, seen = [], set()
        for e in entities:
            if e.id not in seen:
                seen.add(e.id)
                uniq.append(e)
        fact = {
            "category": category,
            "subject": subj.text if subj is not None else (subtype or category),
            "predicate": predicate,
            "object": obj.text if obj is not None else value,
            "scope": scope, "polarity": polarity, "modality": modality, "subtype": subtype,
            "quote": self.sents[sent].text, "entities": uniq, "ambiguous": False,
            "align": {"subject": None, "predicate": pred_key,
                      "object": None if obj is not None else (value_key if value_key is not None else value.lower())},
            "rule": rule, "risk": sorted(set(risk)), "sent_index": sent,
            "spans": sorted({tuple(char_span([t])) for t in toks}),
            "cues": [{"text": c.text, "span": char_span([c])} for c in cues],
            "_subj": subj, "_obj": obj, "_statement": statement,
            "semantic_roles": {name: [r.key() for r in refs if r is not None]
                               for name, refs in (("subject", [subj]), ("object", [obj]),
                                                  ("recipient", list(extra)))},
            "relation_arity": "unary_location" if rule == "absolute_position" else
                              "multi" if extra else "binary" if obj is not None else "unary",
            "voice": "passive" if rule == "passive_agent" else "active_or_unspecified",
        }
        if reverse:
            fact["statement_reversed"] = reverse
        corefs = [r.coref for r in (subj, obj, *extra) if r is not None and r.coref]
        if corefs:
            fact["coref"] = corefs
        alts = [[e.anchor for e in r.entities] for r in (subj, obj) if r is not None and r.alternative]
        if alts:
            fact["alternatives"] = alts
        self.facts.append(fact)
        return fact


    def run(self):


        for t in self.doc:
            if self.is_entity_head(t):
                self.entity_for(t)

        for k, sent in enumerate(self.sents):
            self.sentence(k, list(sent))
        self.ocr()
        return self.finalize()

    def sentence(self, k, tokens):
        for t in tokens:
            if self.is_entity_head(t):
                self.noun_facts(t, k)
        for t in tokens:
            if t.pos_ in {"VERB", "AUX"} and t.i not in self.ocr_tokens:
                if lem(t) in COPULA and t.dep_ not in {"aux", "auxpass"}:
                    self.copula(t, k)
                elif t.pos_ == "VERB":
                    self.verb(t, k)
            elif self.is_gerund_clause(t):
                self.verb(t, k)
        for t in tokens:
            if not self.is_entity_head(t):
                continue
            preps = [p for p in self.preps(t) if p.i not in self.consumed_preps]
            for c in t.children:
                if c.dep_ in {"amod", "acl"} and c.pos_ == "ADJ" and low(c) in L.VISIBILITY_ADJ:
                    self.used.add(c.i)
                    preps.extend(p for p in self.preps(c) if p.i not in self.consumed_preps)


            nneg, nunc, ncues = self.noun_cues(t)
            mod = self.existence_modality(t) if preps else None
            if mod is not None and mod[1] == "uncertain":
                nunc, ncues = True, list(ncues) + [c for c in mod[2] if c not in ncues]
            for p in preps:
                self.consumed_preps.add(p.i)
                for subj in self.ref(t):
                    self.prep_fact(subj, p, k, (nneg, nunc, ncues))
        self.scene_style(k, tokens)


    def existence_modality(self, t):
        """(polarity, modality, cues, risk) for 'X exists', or None if the text makes no such claim."""
        head = self.group_head(t)
        neg, unc, cues = self.noun_cues(t)
        v = self.governing_verb(head)
        vneg, vunc, vcues = self.clause_cues(v)
        role = head.dep_
        existential = v is not None and lem(v) in COPULA and any(c.dep_ == "expl" for c in v.children)


        if role in SUBJ_DEPS and self.visibility_clause(v) and (vneg or vunc):
            if vneg and any(self.has_locative(x) for x in self.visibility_heads(v)):
                return None
            return ("negative" if vneg else "positive"), "uncertain" if (unc or vunc) else "definite", cues + vcues, []
        if neg:
            if self.has_locative(head) or (v is not None and self.has_locative(v)):
                return None
            return "negative", "definite", cues, []
        if role == "pobj":
            risk = ["presupposed_location"] if vneg else []
            return "positive", "uncertain" if (unc or vunc) else "definite", cues + (vcues if vunc else []), risk
        if role in OBJ_DEPS | {"oprd", "dative", "appos"} or (existential and role in {"attr", "nsubj"}):
            if vneg:
                return None
            return "positive", "uncertain" if (unc or vunc) else "definite", cues + (vcues if vunc else []), []

        return "positive", "uncertain" if unc else "definite", cues, (["clause_cue_not_applied"] if (vneg or vunc) else [])

    def visibility_heads(self, v):
        """v 与它的可见性补语（is visible 中的 visible），用于找方位短语。"""
        return [v] + [c for c in v.children if c.dep_ in {"acomp", "oprd"} and low(c) in L.VISIBILITY_ADJ]

    def visibility_clause(self, v):
        """'X is (not) visible / present'、'X cannot be seen / is not shown'。"""
        if v is None:
            return False
        if lem(v) in COPULA and any(c.dep_ in {"acomp", "oprd"} and low(c) in L.VISIBILITY_ADJ for c in v.children):
            return True
        return (lem(v) in L.VISIBILITY_VERBS or low(v) in L.VISIBILITY_ADJ) and v.tag_ == "VBN" \
            and any(c.dep_ in {"auxpass", "nsubjpass"} for c in v.children)

    def has_locative(self, t):
        for c in t.children:
            if c.dep_ == "prep" and low(c) in L.SPATIAL_PREPS:
                return True
            if c.dep_ in {"advmod", "amod"} and low(c) in L.MULTIWORD_HEADS:
                return True
        return False

    def noun_facts(self, t, k):
        head = self.group_head(t)
        members, disj = self.conjuncts(head)
        members = [x for x in members if self.is_entity_head(x)]
        alternative = disj and len(members) > 1
        if alternative and t is not head:
            return
        if not alternative:
            members = [t]
        mod = self.existence_modality(t)
        nneg, nunc, ncues = self.noun_cues(t)
        v = self.governing_verb(head)
        vneg, vunc, _ = self.clause_cues(v)
        for x in members:
            sref = self.single(x)
            if mod is not None:
                pol, moda, cues, risk = mod
                if alternative:
                    moda, risk = "uncertain", list(risk) + ["alternative"]
                self.emit("object", sref, "exists", "exists", polarity=pol, modality=moda,
                          rule="noun_alternative" if alternative else "noun", cues=cues, risk=risk, sent=k)


            uncertain_np = alternative or nunc or mod is None or (mod is not None and mod[1] == "uncertain")
            moda = "uncertain" if uncertain_np else "definite"
            for c in x.children:
                if c.i in self.ocr_tokens or c.i in self.style_tokens:
                    continue
                if c.dep_ == "amod" and self.lexicalized(c, x):
                    self.used.add(c.i)
                elif c.dep_ == "amod":
                    for adj in self.conjuncts(c)[0]:
                        self.attribute(sref, adj, k, moda, x)
                elif c.dep_ == "compound" and lem(c) in L.MATERIALS:
                    self.emit("attribute", sref, "material", "material", value=c.text, value_key=lem(c),
                              modality=moda, rule="compound_material", tokens=[c], sent=k)
                elif c.dep_ == "compound" and (c.pos_ == "PROPN" or c.ent_type_ in {"ORG", "PRODUCT", "GPE", "FAC", "NORP"}):
                    self.emit("identity", sref, "identity", "identity", value=c.text, value_key=c.text.lower(),
                              modality=moda, rule="proper_modifier", tokens=[c], sent=k)
                elif c.dep_ == "nummod":
                    self.count(sref, c, k, moda)
                elif c.dep_ == "det" and low(c) in L.QUANT_DETS:
                    self.emit("count", sref, "count", "count", value=c.text, value_key=low(c),
                              modality=moda, rule="quant_det", tokens=[c], sent=k)
                elif c.dep_ == "poss" and c.pos_ in NOUN_POS and self.is_entity_head(c):
                    for owner in self.ref(c):
                        self.emit("relation", owner, "has", "has", obj=sref, modality=moda,
                                  rule="possessive", risk=["possessive"], tokens=[c], sent=k)

            if any(c.dep_ == "amod" and low(c) == "few" for c in x.children):
                few = next(c for c in x.children if c.dep_ == "amod" and low(c) == "few")
                toks = [few] + [c for c in x.children if c.dep_ == "det" and c.i == few.i - 1]
                self.emit("count", sref, "count", "count", value=self.surface(toks), value_key="a few",
                          modality=moda, rule="a_few", tokens=toks, sent=k)

        if t.dep_ == "pobj" and low(t.head) == "of" and lem(t.head.head) in L.QUANT_NOUNS:
            q = t.head.head
            qtoks = [x for x in q.lefts if x.dep_ in {"det", "nummod", "amod", "quantmod"}] + [q]
            value = self.surface(qtoks)
            self.emit("count", self.single(t), "count", "count", value=value, value_key=number_value(value),
                      modality="uncertain" if (mod is None or mod[1] == "uncertain") else "definite",
                      rule="quant_noun", tokens=qtoks + [t.head], sent=k)
        if lem(head) in L.PLACES and head.dep_ == "pobj" and low(head.head) in PLACE_PREPS:
            toks = self.group_tokens(members)
            self.emit("scene", None, "setting", "setting", value=self.surface(toks),
                      value_key="|".join(sorted(lem(x) for x in members)), subtype="setting",
                      modality="definite" if mod is None or mod[1] == "definite" or alternative else "uncertain",
                      rule="place_noun", tokens=toks, sent=k,
                      risk=["place_as_setting"] + (["alternative"] if alternative else []))

    def attribute(self, sref, adj, k, modality, owner):
        w = low(adj)
        if w in L.HEDGE_ADV or (w in L.NON_ATTRIBUTE_ADJ and w not in L.POSITION_ADJ) or w == "few":
            if w in L.NON_ATTRIBUTE_ADJ:
                self.used.add(adj.i)
            return
        value = self.modifier_text(adj)
        if w in L.SUBJECTIVE:
            self.emit("other", sref, "is", "subjective", value=value, value_key=lem(adj), subtype="subjective",
                      modality=modality, rule="subjective_adj", tokens=[adj], sent=k)
            return
        if w in L.WEATHER and lem(owner) in {"day", "sky", "weather", "afternoon", "morning", "evening"}:
            self.emit("scene", None, "weather", "weather", value=value, value_key=lem(adj), subtype="weather",
                      modality=modality, rule="weather_adj", tokens=[adj], sent=k)
            return
        if w in L.INDOOR:
            self.emit("scene", None, "setting", "setting", value=adj.text, value_key=L.INDOOR[w],
                      subtype="setting", modality=modality, rule="indoor_adj", tokens=[adj], sent=k)
            return
        last = value.lower().split()[-1]
        sub = ("color" if last in L.COLORS else "material" if last in L.MATERIALS else
               "size" if last in L.SIZES else "shape" if last in L.SHAPES else
               "pattern" if last in L.PATTERNS else "position" if last in L.POSITION_ADJ else
               "state" if adj.tag_ in {"VBN", "VBG"} else "property")
        self.emit("attribute", sref, sub, sub, value=value, value_key=value.lower(),
                  modality=modality, rule="amod_" + sub, tokens=[adj], sent=k)

    def count(self, sref, num, k, modality):
        toks = [num]
        for c in num.children:
            if c.dep_ in {"quantmod", "advmod"}:
                toks.extend(c.subtree)
        value = self.surface(toks)
        self.emit("count", sref, "count", "count", value=value, value_key=number_value(value),
                  modality=modality, rule="nummod", tokens=toks, sent=k)


    def subjects(self, v):
        """(participant Refs, passive_with_agent) for verb v."""
        passive = any(c.dep_ in {"nsubjpass", "auxpass"} for c in v.children) or (
            v.tag_ == "VBN" and any(c.dep_ == "agent" for c in v.children))
        if passive:
            for c in v.children:
                if c.dep_ == "agent":
                    pobj = next((g for g in c.children if g.dep_ == "pobj"), None)
                    if pobj is not None:
                        return self.ref(pobj), True
        out = []
        for c in v.children:
            if c.dep_ in SUBJ_DEPS and lem(c) not in L.META_NOUNS:
                out.extend(self.ref(c))
        if not out and any(c.dep_ == "expl" for c in v.children):
            for c in v.children:
                if c.dep_ in {"attr", "dobj"}:
                    out.extend(self.ref(c))
        if not out and v.dep_ in {"acl", "relcl"} and v.head.pos_ in NOUN_POS:
            out.extend(self.ref(v.head))
        explicit = any(c.dep_ in SUBJ_DEPS for c in v.children)
        if not out and not explicit and v.dep_ in {"conj", "advcl", "xcomp", "pcomp", "ccomp"} and v.head is not v:
            h = v.head
            if h.pos_ in {"VERB", "AUX"}:
                inherited, _ = self.subjects(h)
                if not inherited and lem(h) in L.META_VERBS:
                    for c in h.children:
                        if c.dep_ in OBJ_DEPS:
                            inherited.extend(self.ref(c))
                out.extend(inherited)
            elif h.pos_ in NOUN_POS:
                out.extend(self.ref(h))
        return out, False

    def verb_phrase(self, subj, v, prt=()):
        """'is walking' / 'leans' / 'stand' for statements; uses the caption's own verb."""
        plural = subj is not None and subj.plural
        be = "are" if plural else "is"
        words = [v.text] + [p.text for p in prt]
        if v.tag_ == "VBG" or (v.pos_ in NOUN_POS and low(v).endswith("ing")):
            return be + " " + " ".join(words)
        if v.tag_ == "VBN":
            return be + " " + " ".join(words)
        if v.tag_ in {"VBZ", "VBP", "VBD"}:
            return " ".join(words)
        first = v.text if plural else inflect_3sg(v.text)
        return " ".join([first] + words[1:])

    def verb(self, v, k):
        w = lem(v)
        neg, unc, cues = self.clause_cues(v)
        pol, moda = ("negative" if neg else "positive"), ("uncertain" if unc else "definite")
        subs, passive_agent = self.subjects(v)
        children = list(v.children)

        if w in L.HEDGE_RAISING and (any(c.dep_ in {"acomp", "xcomp", "ccomp", "oprd"} for c in children)
                                     or any(c.dep_ == "prep" and low(c) == "like" for c in children) or w != "look"):
            self.used.add(v.i)
            for c in children:
                if c.dep_ in {"acomp", "oprd", "xcomp"} and c.pos_ == "ADJ":
                    for s in subs:
                        for adj in self.conjuncts(c)[0]:
                            self.attribute_value(s, adj, k, pol, "uncertain", cues + [v])
                if c.dep_ == "prep" and low(c) == "like":
                    pobj = next((g for g in c.children if g.dep_ == "pobj"), None)
                    if pobj is not None:
                        toks = self.np_value_tokens(pobj)
                        for s in subs:
                            self.emit("identity", s, "identity", "identity", value=self.surface(toks),
                                      value_key=lem(pobj), polarity=pol, modality="uncertain",
                                      rule="looks_like", tokens=[c] + toks, cues=cues + [v], sent=k)
            return
        if w in L.HEDGE_INFER:
            self.used.add(v.i)
            return
        if w in L.LINKING_VERBS and any(c.dep_ in {"acomp", "oprd"} for c in children):
            self.used.add(v.i)
            subj_tokens = [c for c in children if c.dep_ in SUBJ_DEPS]
            meta = lem(subj_tokens[0]) if subj_tokens else ""
            if subj_tokens and subj_tokens[0].i not in self.mention_entity:
                self.used.add(subj_tokens[0].i)
            for c in children:
                if c.dep_ in {"acomp", "oprd"}:
                    for adj in self.conjuncts(c)[0]:
                        self.copula_acomp(subs, adj, k, pol, moda, cues, meta)
            return
        explicit = [c for c in children if c.dep_ in SUBJ_DEPS]
        if w in L.META_VERBS and (not subs or (explicit and all(lem(c) in L.META_NOUNS for c in explicit))):
            self.used.add(v.i)
            shown = [r for c in children if c.dep_ in OBJ_DEPS for r in self.ref(c)]
            for c in children:
                if c.dep_ in OBJ_DEPS and lem(c) in L.PLACES:
                    self.emit("scene", None, "setting", "setting", value=c.text, value_key=lem(c), subtype="setting",
                              polarity=pol, modality=moda, rule="shown_place", tokens=[c], cues=cues, sent=k,
                              risk=["place_as_setting"])
            for p in self.preps(v):
                for r in shown:
                    self.prep_fact(r, p, k, (neg, unc, cues), verb=None)
            return
        if w == "take" and any(c.dep_ in OBJ_DEPS and lem(c) == "place" for c in children):
            self.used.update([v.i] + [c.i for c in children if c.dep_ in OBJ_DEPS])
            for p in self.preps(v):
                pobj = next((g for g in p.children if g.dep_ == "pobj"), None)
                if pobj is not None and low(p) in PLACE_PREPS:
                    toks = self.np_value_tokens(pobj)
                    self.emit("scene", None, "setting", "setting", value=self.surface(toks), value_key=lem(pobj),
                              subtype="setting", polarity=pol, modality=moda, rule="takes_place",
                              tokens=[p] + toks, cues=cues, sent=k)
                elif pobj is not None and low(p) == "with":
                    continue
            return
        has_recipient = any(c.dep_ in {"dative", "iobj"} or
                            (c.dep_ == "prep" and low(c) in {"to", "for"}) for c in children)
        if w in L.EXPLANATORY_VERBS and not has_recipient:
            self.used.add(v.i)
            objs_ = [r for c in children if c.dep_ in OBJ_DEPS for r in self.ref(c)]
            for s in subs:
                for o in objs_:
                    self.emit("relation", s, v.text, w, obj=o, polarity=pol, modality=moda, rule="explanatory_object",
                              tokens=[v], cues=cues, risk=["explanatory_verb", "direction"], sent=k,
                              statement="%s %s %s." % (cap_np(s), self.verb_phrase(s, v), the_np(o)),
                              reverse="%s %s %s." % (cap_np(o), self.verb_phrase(o, v), the_np(s)))
            return
        if w in L.OCR_VERBS and any(c.i in self.ocr_tokens for c in v.subtree):
            self.used.add(v.i)
            return
        prt = [c for c in children if c.dep_ == "prt"]
        pred_text = " ".join([v.text] + [p.text for p in prt])
        pred_key = " ".join([w] + [low(p) for p in prt])
        if passive_agent:
            objs = [r for c in children if c.dep_ == "nsubjpass" for r in self.ref(c)]
            if not objs:
                if v.dep_ in {"acl", "relcl"} and v.head.pos_ in NOUN_POS:
                    objs = self.ref(v.head)
                elif v.head is not v and v.head.pos_ in {"VERB", "AUX"}:
                    objs, _ = self.subjects(v.head)
        else:
            objs = [r for c in children if c.dep_ in OBJ_DEPS for r in self.ref(c)]
        obj_lemmas = {lem(c) for c in children if c.dep_ in OBJ_DEPS}
        if not subs:
            if objs or any(c.dep_ == "prep" for c in children) or w not in L.STATE_VERBS:
                self.pending.append(self.pending_item(v, "no resolvable participant for '%s'" % v.text,
                                                      [r.text for r in objs]))
            return
        recipients, rtoks = [], []
        for c in children:
            if c.dep_ in {"dative", "iobj"} or (c.dep_ == "prep" and low(c) in {"to", "for"} and objs):
                pobj = next((g for g in c.children if g.dep_ == "pobj"), None) if c.pos_ == "ADP" else c
                if pobj is not None:
                    rs = self.ref(pobj)
                    if rs:
                        recipients.extend(rs)
                        rtoks.extend(([c] if c.pos_ == "ADP" else []) + self.np_value_tokens(pobj))
                        self.consumed_preps.add(c.i)
        scope = ""
        if rtoks:
            first = min(rtoks, key=lambda x: x.i)
            last = max(rtoks, key=lambda x: x.i)
            scope = self.caption[first.idx:last.idx + len(last.text)]
        stative_passive = w in L.STATE_VERBS and not passive_agent
        subjective = w in L.SUBJECTIVE_VERBS or (w in {"share", "have", "spend"} and obj_lemmas & L.SUBJECTIVE_OBJECTS)
        idiom = w == "make" and "way" in obj_lemmas
        for s in subs:
            if idiom:
                way = next(c for c in children if c.dep_ in OBJ_DEPS and lem(c) == "way")
                toks = [v, way] + list(way.lefts)
                self.emit("action", s, self.surface(toks), "make way", polarity=pol, modality=moda,
                          rule="verb_idiom", tokens=toks, cues=cues, sent=k,
                          statement="%s %s." % (cap_np(s), self.verb_phrase(s, v) + " " + self.surface(toks[1:] if len(toks) > 1 else [way])))
            elif subjective:
                objtok = [c for c in children if c.dep_ in OBJ_DEPS]
                value_toks = sorted({x for c in objtok for x in self.np_value_tokens(c)}, key=lambda x: x.i)
                value = self.surface(value_toks) if value_toks else ""
                self.used.update(x.i for x in value_toks if low(x) not in L.WEATHER and low(x) not in L.INDOOR)
                self.emit("other", s, pred_text, pred_key, value=value,
                          value_key=" ".join(lem(x) for x in value_toks if x.pos_ in NOUN_POS) or value.lower(),
                          polarity=pol, modality=moda, subtype="subjective_activity", rule="subjective_verb",
                          tokens=[v] + prt + objtok, cues=cues, sent=k,
                          statement="%s %s%s." % (cap_np(s), self.verb_phrase(s, v, prt), (" " + value) if value else ""))
            elif objs:
                for o in objs:
                    if passive_agent:
                        self.emit("relation", s, pred_text, pred_key, obj=o, polarity=pol, modality=moda,
                                  rule="passive_agent", tokens=[v] + prt, cues=cues, risk=["passive", "direction"], sent=k,
                                  statement="%s %s %s." % (cap_np(s), w if s.plural else inflect_3sg(w), the_np(o)))
                        continue
                    self.emit("relation", s, pred_text, pred_key, obj=o, polarity=pol, modality=moda,
                              rule="passive_agent" if passive_agent else "verb_object", tokens=[v] + prt,
                              cues=cues, scope=scope, extra=recipients,
                              risk=(["passive"] if passive_agent else []) + (["recipient"] if recipients else []) + ["direction"],
                              statement="%s %s %s%s." % (cap_np(s), self.verb_phrase(s, v, prt), the_np(o), (" " + scope) if scope else ""),
                              reverse="%s %s %s%s." % (cap_np(o), self.verb_phrase(o, v, prt), the_np(s), (" " + scope) if scope else ""),
                              sent=k)
            elif w in L.STATE_VERBS or stative_passive:
                self.used.add(v.i)
                if v.tag_ == "VBN" and w not in {"see", "find", "notice", "observe", "locate", "situate", "position", "place", "set"}:
                    self.emit("attribute", s, "state", "state", value=v.text, value_key=w, polarity=pol,
                              modality=moda, rule="state_participle", tokens=[v] + prt, cues=cues, sent=k)
            else:
                rel = [c for c in children if c.dep_ == "prep" and (w, low(c)) in L.RELATION_VERB_PREPS]
                if rel:
                    for p in rel:
                        self.consumed_preps.add(p.i)
                        pobj = next((g for g in p.children if g.dep_ == "pobj"), None)
                        for o in self.ref(pobj):
                            self.emit("relation", s, pred_text + " " + p.text, pred_key + " " + low(p), obj=o,
                                      polarity=pol, modality=moda, rule="verb_prep_relation", tokens=[v, p] + prt,
                                      cues=cues, risk=["direction"], sent=k,
                                      statement="%s %s %s %s." % (cap_np(s), self.verb_phrase(s, v, prt), p.text, the_np(o)),
                                      reverse="%s %s %s %s." % (cap_np(o), self.verb_phrase(o, v, prt), p.text, the_np(s)))
                else:
                    dobj = next((c for c in children if c.dep_ in OBJ_DEPS), None)
                    if dobj is not None:
                        vp_toks = [v] + prt + list(dobj.subtree)
                        a, b = min(x.i for x in vp_toks), max(x.i for x in vp_toks)
                        span = [self.doc[i] for i in range(a, b + 1)]
                        text = self.surface(span)
                        self.emit("action", s, text, pred_key + " " + lem(dobj), polarity=pol, modality=moda,
                                  rule="verb_abstract_object", tokens=span, cues=cues, sent=k,
                                  statement="%s %s%s." % (cap_np(s), self.verb_phrase(s, v, prt),
                                                          self.caption[v.idx + len(v.text):span[-1].idx + len(span[-1].text)]))
                    else:
                        self.emit("action", s, pred_text, pred_key, polarity=pol, modality=moda,
                                  rule="intransitive_verb", tokens=[v] + prt, cues=cues, sent=k,
                                  statement="%s %s." % (cap_np(s), self.verb_phrase(s, v, prt)))
            for p in self.preps(v):
                if p.i in self.consumed_preps:
                    continue
                self.prep_fact(s, p, k, (neg, unc, cues), verb=v, has_object=bool(objs))

    def copula(self, v, k):
        neg, unc, cues = self.clause_cues(v)
        pol, moda = ("negative" if neg else "positive"), ("uncertain" if unc else "definite")
        expl = any(c.dep_ == "expl" for c in v.children)
        self.used.add(v.i)
        subj_tokens = [c for c in v.children if c.dep_ in SUBJ_DEPS]
        if subj_tokens and all(self.is_dummy_it(c, v) for c in subj_tokens):
            self.used.update(c.i for c in subj_tokens)
            subs = []
        elif subj_tokens:
            subs = [r for c in subj_tokens for r in self.ref(c)]
        else:
            subs, _ = self.subjects(v)
        meta = lem(subj_tokens[0]) if subj_tokens else ""
        if meta and subj_tokens[0].i not in self.mention_entity:
            self.used.add(subj_tokens[0].i)
        for c in v.children:
            if c.dep_ == "attr" and not expl:
                self.copula_attr(subs, c, k, pol, moda, cues, meta)
            elif c.dep_ == "acomp":
                for adj in self.conjuncts(c)[0]:
                    self.copula_acomp(subs, adj, k, pol, moda, cues, meta)
        targets = subs
        if expl:
            targets = [r for c in v.children if c.dep_ in {"attr", "dobj", "nsubj"} for r in self.ref(c)]
        vis_preps = [p for c in v.children if c.dep_ == "acomp" and low(c) in L.VISIBILITY_ADJ for p in self.preps(c)]
        if vis_preps:
            self.used.update(c.i for c in v.children if c.dep_ == "acomp" and low(c) in L.VISIBILITY_ADJ)
        for p in self.preps(v) + vis_preps:
            if p.i in self.consumed_preps:
                continue
            for s in targets:
                nn = self.noun_cues(s.tokens[-1]) if s.tokens and s.tokens[-1].pos_ in NOUN_POS else (False, False, [])
                self.prep_fact(s, p, k, (neg or nn[0], unc or nn[1], cues + nn[2]), verb=v)
            if not targets and meta in L.META_NOUNS | {"atmosphere", "weather", "it"}:
                self.prep_scene(p, k, pol, moda, cues)

    @staticmethod
    def is_dummy_it(t, v):
        """天气 / 时间句的形式主语：It is a sunny day. / It is night. / It is dark outside."""
        if low(t) != "it" or lem(v) not in COPULA:
            return False
        return any((c.dep_ == "attr" and lem(c) in L.TIME | {"day"}) or
                   (c.dep_ == "acomp" and low(c) in L.WEATHER | L.LIGHTING) for c in v.children)

    def copula_attr(self, subs, attr, k, pol, moda, cues, meta):
        w = lem(attr)
        items, disj = self.conjuncts(attr)
        toks = self.group_tokens(items)
        value = self.surface(toks)
        key = "|".join(sorted(lem(x) for x in items))
        if meta in L.META_NOUNS and (w in L.STYLE_NOUNS or attr.i in self.style_tokens):
            self.style_phrase(attr, k, pol, moda, cues)
            return
        if meta in {"scene", "setting", "location", "place", "image", "picture", "photo"}:
            self.emit("scene", None, "setting", "setting", value=self.surface([attr]) if len(items) == 1 else value,
                      value_key=key, subtype="setting", polarity=pol, modality=moda, rule="scene_copula",
                      tokens=[attr] + items, cues=cues, sent=k, risk=["alternative"] if disj else [])
            return
        if w in L.PART_NOUNS and self.of_object(attr) is not None:
            whole = self.ref(self.of_object(attr))
            for s in subs:
                for o in whole:
                    self.emit("relation", s, "part of", "part of", obj=o, polarity=pol, modality=moda,
                              rule="part_of", tokens=[attr, self.of_object(attr).head], cues=cues, sent=k,
                              statement="%s %s part of %s." % (cap_np(s), "are" if s.plural else "is", the_np(o)))
            return
        det = [c for c in items[0].lefts if c.dep_ == "det" and c.i < toks[0].i]
        shown = self.surface(det + toks)
        for s in subs:
            self.emit("identity", s, "identity", "identity", value=value, value_key=key, polarity=pol,
                      modality=moda, rule="copula_attr", tokens=toks + det, cues=cues, sent=k,
                      risk=["alternative"] if disj else [],
                      statement="%s %s %s." % (cap_np(s), "are" if s.plural else "is", shown))

    def copula_acomp(self, subs, adj, k, pol, moda, cues, meta):
        w = low(adj)
        if w in L.HEDGE_ADV or lem(adj) in {"likely", "possible", "unclear", "visible", "present"}:
            self.used.add(adj.i)
            return
        if meta in {"atmosphere", "mood", "ambiance", "ambience", "vibe", "feeling"} or (meta == "scene" and w in L.SUBJECTIVE):
            self.emit("other", None, "is", meta or "atmosphere", value=adj.text, value_key=lem(adj), subtype="atmosphere",
                      polarity=pol, modality=moda, rule="atmosphere_acomp", tokens=[adj], cues=cues, sent=k,
                      statement="The %s is %s." % (meta or "atmosphere", adj.text))
            return
        if meta in {"weather", "day", "sky"} and w in L.WEATHER:
            self.emit("scene", None, "weather", "weather", value=adj.text, value_key=lem(adj), subtype="weather",
                      polarity=pol, modality=moda, rule="weather_acomp", tokens=[adj], cues=cues, sent=k)
            return
        if meta in L.META_NOUNS and adj.i in self.style_tokens:

            toks = [self.doc[i] for i in sorted(self.style_tokens) if self.sent_of[i] == k]
            if not all(t.i in self.used for t in toks):
                self.emit("style", None, "style", "style", value=self.surface(toks), value_key="black-and-white",
                          subtype="style", polarity=pol, modality=moda, rule="bw_style", tokens=toks, cues=cues, sent=k)
            return
        if meta in L.META_NOUNS and w in L.STYLE_ADJ:
            self.emit("style", None, "style", "style", value=adj.text, value_key=lem(adj), subtype="style",
                      polarity=pol, modality=moda, rule="style_acomp", tokens=[adj], cues=cues, sent=k)
            return
        for s in subs:
            self.attribute_value(s, adj, k, pol, moda, cues)

    def attribute_value(self, s, adj, k, pol, moda, cues):
        value = self.modifier_text(adj)
        last = value.lower().split()[-1]
        if last in L.SUBJECTIVE:
            self.emit("other", s, "is", "subjective", value=value, value_key=lem(adj), subtype="subjective",
                      polarity=pol, modality=moda, rule="subjective_acomp", tokens=[adj], cues=cues, sent=k)
            return
        sub = ("color" if last in L.COLORS else "material" if last in L.MATERIALS else
               "size" if last in L.SIZES else "shape" if last in L.SHAPES else
               "pattern" if last in L.PATTERNS else "property")
        self.emit("attribute", s, sub, sub, value=value, value_key=value.lower(), polarity=pol, modality=moda,
                  rule="acomp_" + sub, tokens=[adj], cues=cues, sent=k)


    def preps(self, v):


        cut = min((c.i for c in v.children if c.dep_ == "conj" and low(c) in L.PARTITIVE), default=None)
        out = []
        for c in v.children:
            if cut is not None and c.i > cut and c.dep_ == "prep":
                self.used.add(c.i)
                self.pending.append(self.pending_item(c, "phrase belongs to partitive '%s'" % self.doc[cut].text, []))
                continue
            if c.dep_ == "prep":
                out.append(c)
            elif c.dep_ in {"advmod", "amod", "acomp", "prep"} and low(c) in L.MULTIWORD_HEADS:
                out.extend(g for g in c.children if g.dep_ == "prep" and low(g) == L.MULTIWORD_HEADS[low(c)])
        return out

    def prep_phrase(self, p):
        """(predicate_text, key, target_token|None, tokens) or None."""
        tokens = [p]
        prefix_tok = p.head if (low(p.head) in L.MULTIWORD_HEADS and L.MULTIWORD_HEADS[low(p.head)] == low(p)) else None
        if prefix_tok is not None:
            tokens.append(prefix_tok)
        pobj = next((g for g in p.children if g.dep_ == "pobj"), None)
        if pobj is None:
            return None
        if lem(pobj) in L.POSITION_NOUNS:
            inner = [p, pobj] + [x for x in pobj.lefts if x.dep_ in {"det", "amod", "compound"}]
            of = next((g for g in pobj.children if g.dep_ == "prep" and low(g) == "of"), None)
            target = None
            if of is not None:
                target = next((g for g in of.children if g.dep_ == "pobj"), None)
                if target is not None and lem(target) in L.META_NOUNS:
                    self.used.update(x.i for x in [of, target] + list(target.lefts))
                    target = None
                else:
                    inner.append(of)
            all_toks = sorted(set(tokens + inner), key=lambda x: x.i)
            text = self.surface(all_toks)
            key = " ".join(low(x) for x in all_toks if x.dep_ != "det")
            return text, key, target, all_toks
        all_toks = sorted(set(tokens), key=lambda x: x.i)
        text = self.surface(all_toks) if len(all_toks) > 1 else p.text
        return text, " ".join(low(x) for x in all_toks), pobj, all_toks

    def prep_fact(self, subj, p, k, base_cues=(False, False, []), verb=None, has_object=False):
        word = low(p)
        multi = low(p.head) in L.MULTIWORD_HEADS
        neg, unc, cues = base_cues
        pol, moda = ("negative" if neg else "positive"), ("uncertain" if unc else "definite")
        pobj = next((g for g in p.children if g.dep_ == "pobj"), None)
        if word in {"of", "for", "as", "like", "than", "during", "about", "including", "despite",
                    "except", "per", "via", "while", "if", "because"} and not multi:
            return
        if word in {"with", "without"}:
            for o in self.ref(pobj):
                if o.key() == subj.key() or {e.head_key for e in o.entities} & {e.head_key for e in subj.entities}:
                    continue
                self.emit("relation", subj, "with", "with", obj=o,
                          polarity="negative" if (neg or word == "without") else "positive",
                          modality=moda, rule="prep_" + word, tokens=[p],
                          cues=list(cues) + ([p] if word == "without" else []),
                          risk=["with_attachment"], sent=k,
                          statement="%s %s %s." % (cap_np(subj), "have" if subj.plural else "has", the_np(o)))
            return
        if word == "by" and verb is not None and any(c.dep_ == "auxpass" for c in verb.children):
            return
        if word not in L.SPATIAL_PREPS and not multi:
            self.pending.append(self.pending_item(p, "unclassified preposition '%s'" % p.text, [subj.text]))
            return
        phrase = self.prep_phrase(p)
        if phrase is None:
            return
        pred_text, key, target, toks = phrase
        risk = ["verb_attached_prep"] if (verb is not None and has_object) else []
        vp = None


        if verb is not None and lem(verb) not in COPULA and not has_object:
            vp = self.verb_phrase(subj, verb, [c for c in verb.children if c.dep_ == "prt"])
            way = next((c for c in verb.children if c.dep_ in OBJ_DEPS and lem(c) == "way"), None)
            if way is not None:
                vp += " " + self.surface(list(way.lefts) + [way])
        if target is None:
            if key in {"in", "at", "on", "to"}:
                return
            self.emit("spatial", subj, pred_text, key, value="", value_key="", polarity=pol, modality=moda,
                      rule="absolute_position", tokens=toks, cues=cues, risk=risk, sent=k,
                      statement="%s %s %s." % (cap_np(subj), vp or ("are" if subj.plural else "is"), pred_text.lower()))
            return
        if lem(target) in L.META_NOUNS:
            self.used.update(x.i for x in toks + [target])
            return
        if lem(target) in L.BODY_PARTS and has_object and any(c.dep_ == "poss" for c in target.children):
            self.used.update(x.i for x in toks + [target] + [c for c in target.children if c.dep_ == "poss"])
            return
        def rvp(o):
            if verb is not None and lem(verb) not in COPULA and not has_object:
                return self.verb_phrase(o, verb, [c for c in verb.children if c.dep_ == "prt"])
            return "are" if o.plural else "is"
        objs = self.ref(target)
        if not objs:
            if lem(target) in L.TIME:
                self.prep_scene(p, k, pol, moda, cues)
            return


        if word == "in" and not multi and (lem(target) in L.CLOTHING or low(target) in L.CLOTHING):
            for o in objs:
                if o.key() == subj.key():
                    continue
                self.emit("relation", subj, "wear", "wear", obj=o, polarity=pol, modality=moda,
                          rule="prep_in_clothing", tokens=toks, cues=cues, risk=["clothing_in"], sent=k,
                          statement="%s %s wearing %s." % (cap_np(subj), "are" if subj.plural else "is", the_np(o)))
            return
        for o in objs:
            if o.key() == subj.key():
                continue
            self.emit("spatial", subj, pred_text, key, obj=o, polarity=pol, modality=moda,
                      rule="prep_spatial", tokens=toks, cues=cues,
                      risk=risk + ([] if key in L.SYMMETRIC_SPATIAL else ["direction"]), sent=k,
                      statement="%s %s %s %s." % (cap_np(subj), vp or ("are" if subj.plural else "is"), pred_text.lower(), the_np(o)),
                      reverse="%s %s %s %s." % (cap_np(o), rvp(o), pred_text.lower(), the_np(subj)))

    def prep_scene(self, p, k, pol, moda, cues):
        pobj = next((g for g in p.children if g.dep_ == "pobj"), None)
        if pobj is not None and lem(pobj) in L.TIME:
            self.emit("scene", None, "time", "time", value=pobj.text, value_key=lem(pobj), subtype="time",
                      polarity=pol, modality=moda, rule="time_pp", tokens=[p, pobj], cues=cues, sent=k)


    def word_modality(self, t):
        _, unc, _ = self.clause_cues(self.governing_verb(t))
        return "uncertain" if unc else "definite"

    def scene_style(self, k, tokens):
        for t in tokens:
            if t.i in self.used or t.i in self.ocr_tokens or t.i in self.style_tokens:
                continue
            w, lw = low(t), lem(t)
            if w in L.INDOOR and t.pos_ in {"ADV", "ADJ"} and t.dep_ != "prep":
                self.emit("scene", None, "setting", "setting", value=t.text, value_key=L.INDOOR[w], subtype="setting",
                          rule="indoor_word", tokens=[t], sent=k, modality=self.word_modality(t))
            elif lw in L.TIME and t.pos_ == "NOUN" and lw != "day":
                self.emit("scene", None, "time", "time", value=t.text, value_key=lw, subtype="time",
                          rule="time_word", tokens=[t], sent=k, modality=self.word_modality(t))
            elif w in L.WEATHER and t.pos_ == "ADJ" and t.dep_ in {"amod", "acomp"}:

                owner = [t.head] if t.dep_ == "amod" and lem(t.head) in L.TIME | {"day"} else []
                self.emit("scene", None, "weather", "weather", value=t.text, value_key=lw, subtype="weather",
                          rule="weather_word", tokens=[t] + owner, sent=k, modality=self.word_modality(t))
            elif w in {"dimly", "brightly"} or (w == "lit" and t.tag_ == "VBN"):
                toks = sorted([t] + [c for c in t.children if c.dep_ == "advmod"] + ([t.head] if w in {"dimly", "brightly"} and low(t.head) == "lit" else []), key=lambda x: x.i)
                text = self.surface(toks)
                self.emit("scene", None, "lighting", "lighting", value=text, value_key=text.lower(),
                          subtype="lighting", rule="lighting_word", tokens=toks, sent=k, modality=self.word_modality(t))
        style = [t for t in tokens if t.i in self.style_tokens]
        if style and not all(t.i in self.used for t in style):
            self.emit("style", None, "style", "style", value=self.surface(style), value_key="black-and-white",
                      subtype="style", rule="bw_style", tokens=style, sent=k)
        for t in tokens:
            if t.i not in self.used and lem(t) in L.STYLE_NOUNS and t.pos_ == "NOUN" and self.is_image_reference(t):
                self.style_phrase(t, k, "positive", self.word_modality(t), [])

    def style_phrase(self, t, k, pol, moda, cues):
        self.emit("style", None, "medium", "medium", value=t.text, value_key=lem(t), subtype="medium",
                  polarity=pol, modality=moda, rule="style_noun", tokens=[t], cues=cues, sent=k,
                  statement="The image is a %s." % t.text)
        for c in t.children:
            if c.dep_ in {"amod", "compound"} and (low(c) in L.STYLE_ADJ or c.i in self.style_tokens):
                toks = [c] if c.i not in self.style_tokens else [self.doc[i] for i in sorted(self.style_tokens) if self.sent_of[i] == k]
                self.emit("style", None, "style", "style", value=self.surface(toks), value_key=self.surface(toks).lower(),
                          subtype="style", polarity=pol, modality=moda, rule="style_adj", tokens=toks, cues=cues, sent=k,
                          statement="The image is %s." % self.surface(toks))


    def ocr(self):
        for m in OCR_PATTERN.finditer(self.caption):
            literal = next(g for g in m.groups() if g is not None)
            a, b = m.span()
            inside = [t for t in self.doc if a <= t.idx < b]
            if not inside:
                continue
            k = self.sent_of[inside[0].i]
            owner = None
            for t in self.sents[k]:
                if t.dep_ in SUBJ_DEPS and t.i in self.mention_entity and t.i < inside[0].i:
                    owner = self.mention_entity[t.i]
                    break
            if owner is None:
                for t in reversed(list(self.sents[k])):
                    if t.i < inside[0].i and t.i in self.mention_entity:
                        owner = self.mention_entity[t.i]
                        break
            subj = Ref([owner], owner.anchor, []) if owner else None
            neg, unc, cues = self.clause_cues(self.governing_verb(inside[0]))
            self.emit("text_ocr", subj, "text", "text", value=literal, value_key=literal.casefold(),
                      polarity="negative" if neg else "positive", modality="uncertain" if unc else "definite",
                      rule="quoted_literal", tokens=inside, cues=cues, sent=k,
                      risk=[] if owner else ["ocr_owner_unknown"],
                      statement=("%s shows the text \"%s\"." % (cap_np(subj), literal)) if subj else
                                "The image contains the text \"%s\"." % literal)


    SKIP_UNCOVERED = (L.META_NOUNS | L.META_VERBS | L.NON_ATTRIBUTE_ADJ | L.HEDGE_ADV | L.HEDGE_RAISING |
                      L.HEDGE_INFER | L.MODALS | {"be", "have", "can", "also", "there", "here", "addition",
                      "seen", "located", "positioned", "together", "close", "closer", "nearby", "next", "far",
                      "away", "central", "primary", "prominent", "notable", "noticeable", "evident", "clearly",
                      "create", "add", "contribute", "give", "well", "just", "even", "still", "currently",
                      "actively", "way", "part", "number", "variety", "middle", "side", "front", "top",
                      "center", "background", "foreground", "left", "right", "corner", "edge", "distance",
                      "area", "rest", "lot", "one", "total", "group", "couple", "pair", "bunch", "set"})

    def content_tokens(self):
        return [t for t in self.doc if t.pos_ in {"NOUN", "PROPN", "ADJ", "NUM", "VERB"}
                and t.i not in self.ocr_tokens and lem(t) not in self.SKIP_UNCOVERED and low(t) not in self.SKIP_UNCOVERED]

    def uncovered(self):
        out = []
        for t in self.doc:
            if t.i in self.used or t.i in self.ocr_tokens or t.i in self.style_tokens or t.i in self.mention_entity:
                continue
            if t.pos_ not in {"NOUN", "PROPN", "ADJ", "NUM", "VERB"}:
                continue
            if lem(t) in self.SKIP_UNCOVERED or low(t) in self.SKIP_UNCOVERED:
                continue
            k = self.sent_of[t.i]
            out.append({"token": t.text, "pos": t.pos_, "dep": t.dep_, "head": t.head.text,
                        "focus": self.focus_phrase(t),
                        "span": char_span([t]), "sent_index": k, "quote": self.sents[k].text})
        return out

    def sentence_entity_names(self, k, subj):
        names = []
        for e in self.entities:
            if any(self.sent_of[m] == k for m in e.mentions) and e.display:
                names.append("the " + e.display)
        if subj.coref:
            names.append(the_np(subj))
            names += ["the " + a for a in subj.coref.get("alternatives", [])]
        return list(dict.fromkeys(names))


    def finalize(self):
        facts, by_key = [], {}
        for f in self.facts:
            subj, obj, statement = f.pop("_subj"), f.pop("_obj"), f.pop("_statement")
            f["align"]["subject"] = subj.key() if subj is not None else (
                f["category"] if f["category"] in {"scene", "style"} else f["align"]["predicate"] if f["category"] == "other" and subj is None else f["subject"].lower())
            if obj is not None:
                f["align"]["object"] = obj.key()
            f["entities"] = [e.as_dict() for e in f["entities"]]
            for ent in f["entities"]:
                original = next(e for e in self.entities if e.id == ent["id"])
                ent.update(person=original.person, gender=original.gender,
                           source_mentions=[{"span": char_span([self.doc[i]]),
                                             "sent_index": self.sent_of[i],
                                             "quote": self.sents[self.sent_of[i]].text}
                                            for i in original.mentions])
            f["statement"] = statement or render_statement(f, subj, obj)
            f["spans"] = [list(s) for s in f["spans"]]

            if subj is not None and not subj.alternative and len(subj.entities) == 1:
                prefix = cap_np(subj)
                if f["statement"].startswith(prefix + " "):
                    rest = f["statement"][len(prefix):].strip().rstrip(".")
                    if f["polarity"] == "negative":


                        head, _, tail = rest.partition(" ")
                        rest = (head + " not " + tail) if head in {"is", "are"} and tail else None
                if f["statement"].startswith(prefix + " ") and rest:
                    f["role"] = {"question": "which of these " + rest + "?", "answer": the_np(subj),
                                 "sentence_entities": self.sentence_entity_names(f["sent_index"], subj)}
            key = json.dumps(semantic_payload(f), sort_keys=True)
            if key in by_key:
                prior = by_key[key]
                prior.setdefault("mentions", [prior["quote"]])
                if f["quote"] not in prior["mentions"]:
                    prior["mentions"].append(f["quote"])
                prior["risk"] = sorted(set(prior["risk"]) & set(f["risk"]))
                continue
            by_key[key] = f
            facts.append(f)
        groups = {}
        for f in facts:
            if "role" in f:
                g = (f["sent_index"], f["category"], f["align"]["predicate"], f["align"]["object"])
                groups.setdefault(g, []).append(f)
        for members in groups.values():
            answers = {f["role"]["answer"] for f in members}
            for f in members:
                f["role"]["co_subjects"] = sorted(answers - {f["role"]["answer"]})
        uncovered = self.uncovered()
        pending = list(self.pending)
        for u in uncovered:
            pending.append({"quote": u["quote"], "issue": "unparsed content word '%s' (%s)" % (u["token"], u["pos"]),
                            "anchors": [], "span": u["span"], "sent_index": u["sent_index"], "focus": u["focus"]})
        content = self.content_tokens()
        coverage = {"content_words": len(content), "uncovered": len(uncovered),
                    "covered_ratio": round(1 - len(uncovered) / len(content), 4) if content else None}
        return {"caption": self.caption, "facts": facts, "pending": pending, "uncovered": uncovered,
                "content_coverage": coverage,
                "entities": [dict(e.as_dict(), mentions=sorted(e.mentions)) for e in self.entities],
                "sentences": [s.text for s in self.sents]}


def the_np(ref):
    if ref is None:
        return "it"
    if ref.alternative:
        return ref.text
    if len(ref.entities) == 1:
        return "the " + ref.entities[0].display
    return "the " + ref.text


def cap_np(ref):
    s = the_np(ref)
    return s[:1].upper() + s[1:]


def existence_statement(noun, plural):
    """'There is a bench.' / 'There are apples.' / 'There is grass.'（解析器与 ensure_existence 共用）"""
    if plural:
        return "There are %s." % noun
    if noun.split()[-1].lower() in L.MASS_NOUNS:
        return "There is %s." % noun
    first = noun.split()[0].lower()
    if first in {"one", "another", "each"}:
        return "There is %s." % noun
    vowel = noun[:1].lower() in "aeiou" and not re.match(r"(?i)(uni|use|eu|one)", noun)
    return "There is %s %s." % ("an" if vowel else "a", noun)


def render_statement(f, subj=None, obj=None):
    """Positive, definite base form of a claim. Polarity and modality are asked separately."""
    c, val = f["category"], f["object"]
    be = "are" if (subj is not None and subj.plural) else "is"
    if c == "object":
        noun = subj.entities[0].display if (subj is not None and len(subj.entities) == 1) else (subj.text if subj is not None else f["subject"])
        return existence_statement(noun, subj is not None and subj.plural)
    if c == "attribute":
        if f["predicate"] == "position":
            return "%s %s the %s one." % (cap_np(subj), be, val)
        return "%s %s %s." % (cap_np(subj), be, val)
    if c == "count":
        n = number_value(val)
        v = val if val[:1].isdigit() else val[:1].lower() + val[1:]
        of = " of" if v.split()[-1].lower() in L.QUANT_NOUNS else ""
        return "There %s %s%s %s." % ("is" if n == "1" else "are", v, of, subj.text if subj else f["subject"])
    if c == "spatial":
        return "%s %s %s%s." % (cap_np(subj), be, f["predicate"], (" " + the_np(obj)) if obj is not None else "")
    if c == "identity":
        return "%s %s %s." % (cap_np(subj) if subj else "It", be, val)
    if c == "scene":
        return {"time": "It is %s.", "weather": "The weather is %s.", "lighting": "The scene is %s."}.get(
            f["subtype"], "The setting is %s.") % val
    if c == "style":
        return "The image is %s." % val
    if c == "other":
        if subj is None:
            return "The %s is %s." % (f["align"]["predicate"], val)
        return "%s %s %s." % (cap_np(subj), be, val)
    if c in {"action", "relation"}:
        return "%s %s%s." % (cap_np(subj), f["predicate"], (" " + the_np(obj)) if obj is not None else "")
    if c == "text_ocr":
        return "The image contains the text \"%s\"." % val
    return "%s %s %s." % (f["subject"], f["predicate"], val)


def parse_caption(nlp, caption):
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("A nonempty caption is required")
    return CaptionParser(nlp, caption).run()
