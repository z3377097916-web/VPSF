"""Source-grounded pronoun repair; every link needs a closed text-only check."""
import re

ENTITY_PRONOUNS = {"it", "them", "they", "this", "that", "these", "those"}
PRONOUN_RE = re.compile(r"\b(it|them|they|this|that|these|those)\b", re.I)
PLURAL = {"them", "they", "these", "those"}


def _norm_text(text):
    return " ".join(str(text or "").casefold().split())


def _label(e):
    name = e.get("display") or e["anchor"]
    return name if re.match(r"^(the|a|an)\s", name, re.I) else "the " + name


def _evidence(name, sentences, before):
    return [{"sent_index": k, "quote": text} for k, text in enumerate(sentences[:before + 1])
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)]


def resolve(facts, sentences=None):
    sentences = list(sentences or [])
    if not sentences:
        sentences = [""] * (1 + max((f.get("sent_index", 0) for f in facts), default=0))
        for f in facts:
            sentences[f.get("sent_index", 0)] = f.get("quote", "")
    for f in facts:
        ents = f.get("entities") or []
        quote, k = f.get("quote", ""), f.get("sent_index", 0)
        if f.get("category") in {"spatial", "relation"} and len(ents) == 1:

            pred = f.get("predicate", "").strip()
            slot = re.search(r"\b" + re.escape(pred) + r"\s+(it|them|this|that|these|those)\b", quote, re.I) if pred else None
            if slot and (not f.get("object") or f["object"].lower() in ENTITY_PRONOUNS):
                pronoun, candidates = slot.group(1).lower(), {}
                for prev in facts:
                    for e in prev.get("entities") or []:
                        key = e.get("key") or e["anchor"]
                        if key == (ents[0].get("key") or ents[0]["anchor"]):
                            continue
                        if bool(e.get("plural")) != (pronoun in PLURAL) or (pronoun == "it" and e.get("person")):
                            continue
                        for source in _evidence(e["anchor"], sentences, k):
                            idx = source["sent_index"]
                            positions = list(re.finditer(r"(?<!\w)" + re.escape(e["anchor"]) + r"(?!\w)", source["quote"], re.I))
                            positions = [m for m in positions if idx < k or m.end() <= slot.start(1)]
                            if not positions:
                                continue
                            rank = (idx, positions[-1].start())
                            if key not in candidates or rank > candidates[key][0]:
                                candidates[key] = (rank, e, source)
                ranked = sorted(candidates.values(), key=lambda x: x[0], reverse=True)
                if ranked:
                    _, chosen, source = ranked[0]
                    f["entities"].append(dict(chosen))
                    f["object"] = chosen["anchor"]
                    f.setdefault("align", {})["object"] = chosen.get("key") or chosen["anchor"]
                    f.setdefault("semantic_roles", {}).update(subject=[ents[0].get("key") or ents[0]["anchor"]],
                                                               object=[chosen.get("key") or chosen["anchor"]])
                    statement = f.get("statement", "")
                    pattern = re.compile(r"(\b" + re.escape(pred) + r"\s+)" + pronoun + r"\b", re.I)
                    if pattern.search(statement):
                        statement = pattern.sub(lambda m: m.group(1) + _label(chosen), statement, count=1)
                    else:
                        base = statement.rstrip(". ")
                        if not base:
                            base = _label(ents[0]).capitalize() + (" are" if ents[0].get("plural") else " is")
                        statement = base + (" " if base.lower().endswith(pred.lower()) else " " + pred + " ") + _label(chosen) + "."
                    f.update(statement=statement, statement_reversed=None, relation_arity="binary")
                    f["coref"] = [{"pronoun": pronoun, "pronoun_span": [slot.start(1), slot.end(1)],
                                   "span_space": "quote", "antecedent": chosen["anchor"],
                                   "antecedent_id": chosen.get("id"), "evidence": [source],
                                   "alternatives": list(dict.fromkeys(e.get("display") or e["anchor"] for _, e, _ in ranked[1:4])),
                                   "method": "preceding_source_position_and_number"}]
                    prefix = _label(ents[0])
                    if statement.lower().startswith(prefix.lower() + " "):
                        rest = statement[len(prefix):].strip().rstrip(".")
                        if f.get("polarity") == "negative":
                            head, _, tail = rest.partition(" ")
                            rest = head + " not " + tail if head in {"is", "are"} else None
                        if rest:
                            f["role"] = {"question": "which of these " + rest + "?", "answer": prefix,
                                         "sentence_entities": [_label(e) for e in f["entities"]]}
                        else:
                            f.pop("role", None)

        for link in f.get("coref", []):
            link["evidence"] = link.get("evidence") or _evidence(link["antecedent"], sentences, k)
            link["requires_check"] = True
        if f.get("coref"):
            f["risk"] = sorted(set(f.get("risk", [])) | {"coref"})
    return facts


def grounded_antecedents(fact, caption):
    """Only adjudicated links with a source quote license an outside-quote anchor."""
    if not caption or (fact.get("adjudication") or {}).get("status") != "accepted":
        return set()
    checks, names = fact["adjudication"].get("checks", []), set()
    for i, link in enumerate(fact.get("coref", [])):
        passed = any(c.get("kind") == "coref" and c.get("coref_index") == i
                     and c.get("answer") == "antecedent" and c.get("validated") for c in checks)
        name = link["antecedent"]
        caption_norm = _norm_text(caption)
        sourced = any(e.get("quote") and _norm_text(e["quote"]) in caption_norm
                      and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", e["quote"], re.I)
                      for e in link.get("evidence", []))
        if passed and sourced:
            names.add(name.lower())
    return names
