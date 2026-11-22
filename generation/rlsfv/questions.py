"Compact, entity-centred question list (ledger -> guided prompt)."
from __future__ import annotations

from .config import CATEGORIES
from .io import fingerprint
from .prompts import _mention_comparison

GUIDED_COMPACT = (
    "Describe this image in one coherent paragraph using only what is visible. "
    "The items listed below come from earlier descriptions: they are unverified hints, may be wrong, "
    "and are not complete. First check whether each item is really present. Describe only items you can see; "
    "do not mention items that are absent or unclear, and do not invent details. If an item is visible but a "
    "detail is unclear, still describe the item. Report counts, colors, positions, actions and interactions "
    "that you can see, including things not listed. Write the description only, with no lists, questions, "
    "or yes/no answers. Do not use headings or Markdown formatting.\n")


GUIDED_VERIFIED = (
    "Describe this image in one complete paragraph. The questions below are hypotheses, not facts. "
    "Check existence separately from color or appearance. "
    "Check the image: if an item is absent or unclear, do not assert it; if only a detail is unclear, "
    "keep the visible item without guessing that detail. Write only the complete description, not "
    "questions, a list, JSON or yes/no answers. Do not use headings or Markdown formatting.\n")

GUIDED_CLEAN_V3P = (
    "Describe this image in one coherent factual paragraph of 3 to 4 sentences (about 55 to 75 words) "
    "using only clearly visible facts. First check the candidate items and details below against the image, "
    "and naturally include every item, action, and spatial relation you can clearly verify, along with any "
    "other clearly visible foreground or background objects. Do not guess uncertain details, never use 'or' "
    "alternatives, never mention absent items (do not write 'no ...' or 'without ...'), and avoid speculative "
    "summary filler at the end. Write only the factual paragraph with no lists, headings, or yes/no answers.\n")


def check_support(f):
    "Image-verification support of one ledger fact."
    from .scoring import verified, effective_statuses, support
    if not verified(f):
        return None
    return support(effective_statuses(f))


def verified_facts(facts, min_support=1.0, require_definite=True, require_positive=True, min_singleton_p=0.0):
    "Facts allowed into a guided prompt: image-verified, definite and positive."
    out = {}
    for key, f in facts.items():
        support = check_support(f)
        if support is None or support <= 0 or support < min_support:
            continue
        if require_definite and f.get("modality") != "definite":
            continue
        if require_positive and f.get("polarity") != "positive":
            continue
        if min_singleton_p > 0.0 and sum(f.get("counts") or [2, 0, 0]) <= 1:
            pts = [
                float(ch["p_true"])
                for ch in f.get("checks", [])
                if isinstance(ch.get("p_true"), (int, float)) and not ch.get("low_mass") and ch.get("valid", True)
            ]
            if len(pts) == 2 and min(pts) < min_singleton_p:
                continue
        out[key] = f
    return out


RELATION_MODULE = (
    "Describe salient spatial relations, interactions and part-whole relations only when their participants "
    "and the relation itself are clearly visible. Keep who does what to whom and the direction explicit. "
    "Do not enumerate all entity pairs. An unclear relation does not remove otherwise visible objects.\n")


OPEN_ELEVEN = ("Then look for anything not listed: other objects, including small and background ones; "
               "their colors, materials, sizes, shapes and condition; how many there are; where they are "
               "relative to each other; actions and postures; interactions between people, animals and objects; "
               "any legible text or signs; what kind of person, place or brand something is; the setting "
               "(type of place, indoor or outdoor, weather, lighting, time of day); the image style; "
               "and any other clearly visible detail.")

OPEN_SHORT = ("Also include other clearly visible information not listed, including small and background objects. "
              "Do not guess unclear details.")


def entity_key(e):
    return e.get("key") or e.get("head") or e["anchor"].lower()


def entity_name(e):
    return e.get("display") or e["anchor"]


def _role_entities(f, roles):
    """Entities named by semantic roles; handles "A or B" Ref keys joined with '|'."""
    ents = f.get("entities") or []
    out, seen = [], set()
    for role in roles:
        for key in (f.get("semantic_roles") or {}).get(role) or []:
            parts = {key} if "|" not in key else set(key.split("|"))
            for e in ents:
                ek = e.get("key") or e.get("anchor")
                if (ek in parts or e.get("anchor") in parts) and id(e) not in seen:
                    seen.add(id(e)); out.append(e)
    return out


def _counterpart_label(f, include_recipient=False):
    "Label(s) for the non-subject participant(s), driven by semantic roles."
    if f.get("category") not in {"spatial", "relation"}:
        return None
    roles = ["object"] + (["recipient"] if (include_recipient and f.get("category") == "relation") else [])
    ents = _role_entities(f, roles)
    if not ents:
        all_ents = f.get("entities") or []
        ents = all_ents[1:] if f.get("category") == "relation" else all_ents[1:2]
    if not ents:
        return None
    return " and ".join("the " + entity_name(e) for e in ents)


def conditional_detail(f):
    "Every question names its source subject and conditions details on existence."
    from .semantics import unary_location
    subjects = _role_entities(f, ["subject"]) or (f.get("entities") or [])[:1]
    role_keys = (f.get("semantic_roles") or {}).get("subject") or []
    joiner = " or " if any("|" in k for k in role_keys) else " and "
    label = joiner.join("the " + entity_name(e) for e in subjects) or str(f["subject"])
    phrase = claim_phrase_open(f)
    counterpart = _counterpart_label(f, include_recipient=True)
    if f["category"] in {"spatial", "relation"} and not unary_location(f) and counterpart:
        return "If %s and %s are visibly present, %s" % (label, counterpart, phrase)
    plural = len(subjects) > 1 or any(e.get("plural") for e in subjects)
    return "If %s %s visibly present, %s" % (label, "are" if plural else "is", phrase)


def claim_phrase(f):
    """Neutral short check for one claim; never reveals polarity/modality."""
    c, obj, pred = f["category"], f["object"], f["predicate"]
    target_label = _counterpart_label(f, include_recipient=False)
    if c == "attribute":
        sub = f.get("subtype") or pred
        return (sub if sub in {"color", "material", "size", "shape", "pattern"} else None), obj
    if c == "count":
        return "count", obj.lower() if not obj[:1].isdigit() else obj
    if c == "action":
        return None, pred
    if c == "spatial":
        target = target_label or obj
        return None, (pred.lower() + (" " + target if target else "")).strip()
    if c == "relation":
        target = target_label or obj
        scope = (" " + f["scope"]) if f.get("scope") else ""
        return None, (pred + " " + target + scope).strip()
    if c == "identity":
        return "is it", obj
    if c == "text_ocr":
        return "text", '"%s"' % obj
    if c == "other":
        if f.get("subtype") == "subjective_activity":
            return None, (pred + " " + obj).strip()
        return None, obj
    return None, obj or pred


def claim_phrase_open(f):
    "Open question for one claim; never suggests a candidate value."
    c, obj, pred = f["category"], f["object"], f["predicate"]
    target_label = _counterpart_label(f, include_recipient=True)
    if c == "attribute":
        sub = f.get("subtype") or pred
        if sub in {"color", "material", "size", "shape", "pattern"}:
            return "what %s?" % sub
        return ("what is its %s?" % sub) if sub else "what is it like?"
    if c == "count":
        return "how many?"
    if c == "action":
        return "what is it doing?"
    if c == "spatial":
        target = target_label or obj
        return ("where is it relative to %s?" % target) if target else "where is it?"
    if c == "relation":
        target = target_label or obj
        return ("what relation, if any, is visible with %s?" % target) if target else "what relation, if any, is visible?"
    if c == "identity":
        return "what kind is it?"
    if c == "text_ocr":
        return "what legible text is visible?"
    if c == "other":
        return "what is it like?" if not obj else "what is it like?"
    return ("what is its %s?" % pred) if pred else "what is it like?"


def _validate_open_question(rec, facts):
    "Program-level guard: an open relation question must name both participants."
    text = rec.get("text", "")
    name, _, rest = text.partition(": ")
    if not name or not rest:
        raise ValueError("OPEN_QUESTION_RENDER: missing entity or body: %r" % text[:80])


    from .semantics import unary_location
    for claim_id in rec.get("claim_ids", []):
        f = facts.get(claim_id)
        if not f or f.get("category") not in {"spatial", "relation"}:
            continue
        ents = f.get("entities") or []
        if unary_location(f):
            continue
        if len(ents) < 2:
            raise ValueError("OPEN_QUESTION_RENDER: relation fact without counterpart: %r" % text[:120])


        roles = f.get("semantic_roles") or {}
        if roles:
            for role in ("subject", "object", "recipient"):
                role_names = set()
                for key in roles.get(role) or []:
                    parts = {key} if "|" not in key else set(key.split("|"))
                    for e in ents:
                        if (e.get("key") or e.get("anchor")) in parts or e.get("anchor") in parts:
                            role_names.update(n.strip().lower() for n in
                                              (e.get("display"), e.get("anchor"), e.get("key")) if n)
                if role_names and not any(name in text.lower() for name in role_names):
                    raise ValueError("OPEN_QUESTION_RENDER: relation question omitted a participant (%s): %r"
                                     % (role, text[:160]))
        else:
            for e in ents[:2]:
                names = [(e.get(key) or "").strip().lower() for key in ("display", "anchor", "key")]
                names = [name for name in names if name]
                if names and not any(name in text.lower() for name in names):
                    raise ValueError("OPEN_QUESTION_RENDER: relation question omitted a participant: %r" % text[:160])


def compact_checklist(facts, candidates=None, degrade=0, include_pending=True,
                      open_detail=False, max_entities=None, max_details=None,
                      open_observation="full", relation_priority=False, max_detail_questions=None,
                      clean_entities=False, refutation_priority=False):
    "Return question records (same schema as prompts.checklist) with compact text."
    if clean_entities:
        from .prompt_refinement import prompt_view
        view, audit = prompt_view(facts)
        questions = compact_checklist(
            view, candidates, degrade=degrade, include_pending=include_pending,
            open_detail=open_detail, max_entities=max_entities, max_details=max_details,
            open_observation=open_observation, relation_priority=relation_priority,
            max_detail_questions=max_detail_questions, clean_entities=False,
            refutation_priority=refutation_priority)
        for entry in audit["omitted"]:
            questions.append({"question_id": "abstract-" + entry["claim_id"],
                "claim_ids": [entry["claim_id"]], "entity_ids": [], "check_type": ["audit"],
                "text": facts[entry["claim_id"]].get("statement", ""),
                "render": "omitted_prompt_cleaning", "omission_reason": entry["reason"],
                "section": "detail", "eligible_for_scoring": False})
        questions.append({"question_id": "prompt-refinement-audit", "claim_ids": [],
            "entity_ids": [], "check_type": ["audit"], "text": "", "render": "audit_only",
            "section": "detail", "prompt_refinement": audit})
        assert set(facts) <= {k for q in questions for k in q["claim_ids"]}
        return questions
    from .prompt_refinement import needs_recheck
    recheck = {k for k, f in facts.items() if refutation_priority and needs_recheck(f)}
    questions, entities, owned, scene = [], {}, {}, {}
    for key, f in sorted(facts.items()):
        ents = f.get("entities") or []
        for e in ents:
            k = entity_key(e)
            entities.setdefault(k, {"e": e, "claims": set()})["claims"].add(key)
        if f["category"] == "object":
            continue
        if f["category"] in {"scene", "style"} or (f["category"] == "other" and not ents):
            scene.setdefault(f["category"], []).append(key)
        elif ents:
            owned.setdefault(entity_key(ents[0]), []).append(key)
        else:
            scene.setdefault("other", []).append(key)
    order = sorted(entities, key=lambda k: (-len(entities[k]["claims"]), k))
    mentioned_by = {k: {cid for key in e["claims"] for cid in facts[key].get("candidates", [])}
                    for k, e in entities.items()}
    if candidates is not None:
        universe = {c["id"] for c in candidates}
        mentioned_by = {k: ids & universe for k, ids in mentioned_by.items()}


    rel_entities = set()
    for f in facts.values():
        if f.get("category") in {"spatial", "relation"}:
            for e in f.get("entities") or []:
                rel_entities.add(entity_key(e))
    def recheck_entity(k):
        return 0 if entities[k]["claims"] & recheck else 1
    if relation_priority:
        cap_order = sorted(order, key=lambda k: (0 if k in rel_entities else 1,
                                                 recheck_entity(k),
                                                 len(mentioned_by.get(k, ())), -len(entities[k]["claims"]), k))
    else:
        cap_order = sorted(order, key=lambda k: (recheck_entity(k), len(mentioned_by.get(k, ())), -len(entities[k]["claims"]), k))
    kept_order, dropped = list(cap_order if refutation_priority else order), set()
    if max_entities and len(kept_order) > max_entities:
        kept_order = cap_order[:max_entities]
        dropped |= {key for k in cap_order[max_entities:] for key in entities[k]["claims"]}
    for k in kept_order:
        e = entities[k]
        questions.append({"question_id": "exist-" + fingerprint(k)[:16], "claim_ids": sorted(e["claims"]),
                          "entity_ids": [e["e"].get("id", "")], "check_type": ["existence", "omission"],
                          "text": entity_name(e["e"]), "render": "existence"})
    detail_records = []
    for k in [x for x in order if x in set(kept_order)]:
        keys = owned.get(k, [])
        if not keys:
            continue
        detail_items = []
        if open_detail:
            part_map = {}
            def part_priority(key):
                f = facts[key]
                mentions = set(f.get("candidates", []))
                if candidates is not None:
                    mentions &= universe
                return (0 if relation_priority and f["category"] in {"spatial", "relation"} else 1,
                        0 if key in recheck else 1, len(mentions), key)
            for key in sorted(keys, key=part_priority):
                f = facts[key]
                phrase = conditional_detail(facts[key])


                if f["category"] in {"spatial", "relation"} and any(
                        entity_key(e) not in kept_order for e in f.get("entities", [])):
                    questions.append({"question_id": "omitted-participants-" + key,
                                      "claim_ids": [key], "entity_ids": [], "check_type": ["detail"],
                                      "text": phrase, "render": "omitted_by_cap",
                                      "omission_reason": "participant_outside_entity_cap"})
                    continue
                if phrase:
                    part_map.setdefault(phrase, []).append(key)
            detail_items = [{"text": text, "claim_ids": sorted(ids)} for text, ids in part_map.items()]
            parts = list(part_map)
        else:
            groups, plain = {}, []
            for key in keys:
                label, phrase = claim_phrase(facts[key])
                if not phrase:
                    continue
                if label:
                    groups.setdefault(label, [])
                    if phrase not in groups[label]:
                        groups[label].append(phrase)
                elif phrase not in plain:
                    plain.append(phrase)
            parts = ["%s %s?" % (label, " / ".join(v)) for label, v in groups.items()] + [p + "?" for p in plain]
            parts = ["If it and any named participants are visibly present, " + p for p in parts]
            detail_items = [{"text": text, "claim_ids": sorted(keys)} for text in parts]
        if not parts:
            continue
        single = degrade >= 2 and len(mentioned_by.get(k, ())) <= 1
        detail_records.append(((0 if k in rel_entities else 1) if relation_priority else 0,
                               len(mentioned_by.get(k, ())), k,
                               {"question_id": "detail-" + fingerprint(k)[:16],
                                "claim_ids": sorted({key for item in detail_items for key in item["claim_ids"]}),
                                "entity_ids": [entities[k]["e"].get("id", "")],
                                "check_type": ["detail", "omission"],
                                "detail_items": detail_items,
                                "text": entity_name(entities[k]["e"]) + ": " + " ".join(parts),
                                "render": "omitted_budget_existence_only" if single else "detail"}))
    if relation_priority or refutation_priority:
        detail_records.sort(key=lambda item: (item[0], recheck_entity(item[2]), item[1], item[2]))
    detail_records = [item[-1] for item in detail_records]
    if open_detail:
        for rec in detail_records:
            _validate_open_question(rec, facts)
    questions.extend(detail_records)
    labels = {"scene": "Setting", "style": "Image style", "other": "Overall impression"}
    open_labels = {"scene": "If the setting is visibly identifiable, what is it?",
                   "style": "If the image style is clearly identifiable, what is it?",
                   "other": "If other details are clearly visible, what are they?"}
    for cat in ("scene", "style", "other"):
        keys = scene.get(cat, [])
        if not keys:
            continue
        if open_detail:
            text = open_labels[cat]
        else:
            phrases = []
            for key in keys:
                _, phrase = claim_phrase(facts[key])
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
            text = labels[cat] + ": " + " / ".join(phrases) + "?"
        questions.append({"question_id": "global-" + cat, "claim_ids": sorted(keys), "entity_ids": [],
                          "check_type": ["detail"], "text": text,
                          "render": "omitted_budget" if (degrade >= 1 and cat == "other") else "global"})


    lines_used = questions_used = 0
    capped = []
    for rec in questions:
        if rec.get("render") not in {"detail", "global"}:
            capped.append(rec)
            continue
        if max_details is not None and lines_used >= max_details:
            capped.append(dict(rec, render="omitted_by_cap", omission_reason="detail_line_cap"))
            continue
        items = rec.get("detail_items") or [{"text": rec["text"], "claim_ids": rec["claim_ids"]}]
        left = len(items) if max_detail_questions is None else max(0, max_detail_questions - questions_used)
        retained, omitted = items[:left], items[left:]
        if not retained:
            capped.append(dict(rec, render="omitted_by_cap", omission_reason="detail_question_cap"))
            continue
        if omitted:
            capped.append({"question_id": rec["question_id"] + "-omitted-questions",
                           "claim_ids": sorted({k for item in omitted for k in item["claim_ids"]}),
                           "entity_ids": rec["entity_ids"], "check_type": ["detail"],
                           "text": " ".join(item["text"] for item in omitted), "detail_items": omitted,
                           "render": "omitted_by_cap", "omission_reason": "detail_question_cap"})
        if rec.get("render") == "detail":
            prefix = rec["text"].partition(": ")[0]
            rec = dict(rec, detail_items=retained,
                       claim_ids=sorted({k for item in retained for k in item["claim_ids"]}),
                       text=prefix + ": " + " ".join(item["text"] for item in retained))
            if open_detail:
                _validate_open_question(rec, facts)
        rec["detail_question_count"] = len(retained)
        lines_used += 1
        questions_used += len(retained)
        capped.append(rec)
    questions = capped


    known = {" ".join(str(f.get("statement", "")).lower().rstrip(".").split()) for f in facts.values()}
    known_keys = {tuple(f.get("align", {}).values()) + (f["category"],) for f in facts.values() if f.get("align")}
    seen, clues = set(), []
    for c in (candidates if include_pending else []) or []:
        for item in c.get("pending_queries", []):
            if not item.get("visual", True):
                continue
            if " ".join(str(item.get("focus", "")).lower().rstrip(".").split()) in known:
                continue
            if item.get("align_key") and tuple(item["align_key"]) in known_keys:
                continue
            focus = (item.get("focus") or item.get("quote", "")).strip()
            if not focus or focus.lower() in seen:
                continue
            seen.add(focus.lower())
            clues.append((focus, c["id"]))
    if clues:
        questions.append({"question_id": "pending-clues", "claim_ids": [], "entity_ids": [],
                          "check_type": ["detail", "unresolved_source"], "eligible_for_scoring": False,
                          "source_candidates": sorted({cid for _, cid in clues}),
                          "text": "Also check: " + "; ".join(f + "?" for f, _ in clues),
                          "render": "omitted_budget" if degrade >= 1 else "clues"})
    questions.append({"question_id": "open", "claim_ids": [], "entity_ids": [],
                      "check_type": ["open_observation"], "text": OPEN_ELEVEN if open_observation == "full" else OPEN_SHORT,
                      "render": "open", "categories": list(CATEGORIES)})
    if dropped:
        questions.append({"question_id": "omitted-by-cap", "claim_ids": sorted(dropped),
                          "entity_ids": [], "check_type": ["omitted_by_cap"],
                          "eligible_for_scoring": False,
                          "text": "(%d claims omitted to keep the guided prompt short)" % len(dropped),
                          "render": "omitted_by_cap", "omission_reason": "entity_cap"})
    covered = {k for q in questions for k in q["claim_ids"]}
    missing = set(facts) - covered
    assert not missing, "every cumulative fact must map to a question: %s" % sorted(missing)[:3]
    for q in questions:
        q["recheck_claim_ids"] = sorted(set(q["claim_ids"]) & recheck)
        q["section"] = "existence" if "existence" in q["check_type"] else "detail"
        q["comparison"] = _mention_comparison(q["claim_ids"], facts, candidates) if q["claim_ids"] else None
    return questions

def guided_compact(questions, style="compact", relation_module=False):
    exist = [q["text"] for q in questions if q.get("render") == "existence"]
    details = [q["text"] for q in questions if q.get("render") == "detail"]
    tail = [q["text"] for q in questions if q.get("render") in {"global", "clues", "open"}]
    if style == "v3p_clean":
        text = GUIDED_CLEAN_V3P
        if relation_module:
            text += RELATION_MODULE
        if exist:
            text += "Candidate items to check and include if clearly visible: " + ", ".join(exist) + ".\n"
        if details:
            text += "Detail and relation checks (include only clearly visible details and relations): " + " | ".join(details) + "\n"
        return text + "\n".join(tail)
    text = GUIDED_VERIFIED if style in {"verified", "conditional"} else GUIDED_COMPACT
    if relation_module:
        text += RELATION_MODULE
    if exist:
        text += "Existence checks: Is each queried item visibly present, absent, or unclear? " + ", ".join(exist) + ".\n"
    if details:
        text += "Detail checks: Only for visibly present items; relations require all participants to be visible. " + " | ".join(details) + "\n"
    return text + "\n".join(tail)


def estimate_tokens(text):
    """Rough LLaMA token estimate (~3.6 chars/token) for planning only; the backend
    raises ContextOverflow with the real tokenizer."""
    return int(len(text) / 3.6) + 1
