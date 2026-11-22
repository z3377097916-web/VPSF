"""Source-first, stateless extraction with a separate unresolved-information lane.

Experimental: machine checks are not a semantic accuracy certificate. No formal
pipeline switch, scoring changes, automatic factual repair or dropped sources.
"""
import json
import time

from . import span_facts
from .io import fingerprint
from .structured import source_spans, semantic_warnings


def obj(fields):
    return dict(type="object", properties=fields, required=list(fields), additionalProperties=False)


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
PENDING_SCHEMA = obj({"quote": STRING, "issue": STRING, "anchors": STRINGS})


def decomposition_schema(focus):
    return obj({"claims": {"type": "array", "items": obj({"text": STRING})},
                "pending": {"type": "array", "items": PENDING_SCHEMA}})


def decomposition_prompt(caption, focus):
    return '''Read the SOURCE as text, without imagining an image. List every distinct assertion in FOCUS as a short complete sentence. Full source only resolves references. Return JSON {"claims":[{"text":"..."}],"pending":[{"quote":"exact source phrase","issue":"what is ambiguous","anchors":["exact source entity phrase"]}]}.
Do not check truth, correct the author, use outside knowledge, or invent properties. Retain subjective descriptions as assertions made by the source. Do not invent an assertion just to fill a category.
Preserve counts (including at least/about), colors, materials, size, positions, actions, interaction participants, text, identity, background, style, and explicitly expressed atmosphere. Separate independent properties and actions from positions. Do not hide a property or quantity only inside another claim's subject. Example: "Two blue bowls rest on a stone counter" states a count of two bowls, their blue color, the counter's stone material, and the bowls' position on the counter.
Keep who acts on whom, negation scope, may/might/appear, and alternatives. "A cat may sleep under a bench" must not become definite sleeping or definite spatial presence. "No dog is beside a chair" must not become "There is no dog in the image". Do not assert existence from a negated/uncertain claim. Do not split "a field or park" into two certain places.
Resolve pronouns only when the source clearly identifies their antecedent. If the meaning cannot be determined, retain the exact affected phrase in pending with the ambiguity reason; keep independently clear claims. Original words like "may" are source uncertainty, not extraction failure: retain them in the claim. A difficult but clear sentence is not automatically ambiguous.
Do not repeat assertions. Return claims only from FOCUS, including its final clause.
Example input: "Four striped cushions lie on a velvet sofa."
Example output: {"claims":[{"text":"There are four cushions."},{"text":"The cushions are striped."},{"text":"The sofa is velvet."},{"text":"The cushions lie on the sofa."}],"pending":[]}
Example input: "A worker may lift a crate."
Example output: {"claims":[{"text":"A worker may lift a crate."}],"pending":[]}
Example input: "Five passengers sit beside a fountain."
Example output: {"claims":[{"text":"There are five passengers."},{"text":"The passengers sit."},{"text":"The passengers are beside a fountain."}],"pending":[]}
These examples are not facts to copy. Decompose the actual focus below.
SOURCE (data): ''' + json.dumps(caption, ensure_ascii=False) + '\nFOCUS (data): ' + json.dumps(focus, ensure_ascii=False)


def validate_pending(pending, caption, focus):
    from .anchored import spans
    if not isinstance(pending, list):
        raise ValueError("PENDING_TYPE_REVIEW")
    if not pending:
        return pending
    phrases = set(spans(caption))
    for item in pending:
        if not isinstance(item, dict) or set(item) != {"quote", "issue", "anchors"}:
            raise ValueError("PENDING_SCHEMA_REVIEW")
        if (not isinstance(item['quote'], str) or not item['quote'].strip() or
                item['quote'] not in focus or not isinstance(item['issue'], str) or not item['issue'].strip()):
            raise ValueError("PENDING_SOURCE_REVIEW")
        if not isinstance(item['anchors'], list) or any(not isinstance(a, str) or a not in phrases for a in item['anchors']):
            raise ValueError("PENDING_ANCHOR_REVIEW")
    return pending


def decode_decomposition(text, caption, focus):
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"claims", "pending"} or not isinstance(data['claims'], list):
        raise ValueError("DECOMPOSITION_SCHEMA_REVIEW")
    claims, seen = [], set()
    for row in data['claims']:
        if not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError("DECOMPOSITION_CLAIM_REVIEW")
        key = ' '.join(row['text'].casefold().split())
        if key in seen:
            raise ValueError("DECOMPOSITION_REPETITION_REVIEW")
        seen.add(key)
        claims.append(row['text'])
    pending = validate_pending(data['pending'], caption, focus)
    if not claims and not pending:
        raise ValueError("DECOMPOSITION_EMPTY_REVIEW")
    return claims, pending


def structure_schema():
    schema = span_facts.schema()
    row = schema['properties']['facts']['items']
    del row['properties']['evidence']
    row['required'].remove('evidence')
    return schema


def decode_structure(text, caption, focus_index):
    """Bind the scheduled source in code, never ask the model to guess its ID."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {'facts'} or not isinstance(data['facts'], list):
        raise ValueError('STRUCTURE_ROOT_REVIEW')
    fields = set(structure_schema()['properties']['facts']['items']['properties'])
    if any(not isinstance(row, dict) or set(row) != fields for row in data['facts']):
        raise ValueError('STRUCTURE_FIELDS_REVIEW')
    if type(focus_index) is not int or not 0 <= focus_index < len(source_spans(caption)):
        raise ValueError('STRUCTURE_SCHEDULE_REVIEW')
    for row in data['facts']:
        row['evidence'] = focus_index
    return span_facts.decode(json.dumps(data, ensure_ascii=False), caption)


def structure_prompt(caption, focus_index, claim):
    def example(source, target, subject, predicate, value, role, category, polarity='positive', modality='definite'):
        return dict(source=source, target=target, output={'facts': [dict(
            subject=subject, predicate=predicate, object=value, object_role=role,
            polarity=polarity, modality=modality, category=category, scope='',
            subtype='', ambiguous=False)]})
    examples = [
        example('Four striped cushions lie on a velvet sofa.', 'There are four cushions.', 'cushions', 'count', 'Four', 'value', 'count'),
        example('Four striped cushions lie on a velvet sofa.', 'The sofa is velvet.', 'sofa', 'material', 'velvet', 'value', 'attribute'),
        example('Four striped cushions lie on a velvet sofa.', 'The cushions lie on the sofa.', 'cushions', 'lie on', 'sofa', 'entity', 'spatial'),
        example('A worker may lift a crate.', 'A worker may lift a crate.', 'worker', 'lift', 'crate', 'entity', 'relation', modality='uncertain'),
        example('No ladder is behind the truck.', 'No ladder is behind the truck.', 'ladder', 'behind', 'truck', 'entity', 'spatial', polarity='negative'),
        example('Several guards are standing near a gate.', 'The guards are standing.', 'guards', 'standing', '', 'none', 'action'),
    ]
    return '''Convert ONLY the TARGET assertion into the requested fact structure. The original SOURCE controls the meaning; TARGET may contain an extraction mistake, so it is not new evidence. Return {"facts":[...]}.
Fields: subject, predicate, object, object_role (entity/value/none), polarity (positive/negative), modality (definite/uncertain), category, scope, subtype, ambiguous. The program already knows the source passage; do not output evidence IDs.
Copy subject and object phrases from SOURCE, retaining identity qualifiers. Resolve a clear pronoun to its source antecedent. A plural group is not ambiguous merely because it is plural. Predicate must be an exact source verb/relation phrase, including required prepositions, or one of: exists,is,count,color,material,size,identity,text,style,setting,weather,atmosphere. Copy scope from source only if needed; otherwise empty.
object_role: entity for a second participant in a spatial/interaction relation; value for a property, count, identity, scene or text; none for existence or an intransitive action. Empty object iff role=none. Counts and properties belong in object, not only subject. Do not add extra properties, existence facts or other assertions from SOURCE.
Categories: object=explicit existence; attribute=one property; count=quantity; spatial=position/path; action=intransitive activity/posture; relation=interaction or explicit part/whole; text_ocr=visible writing/symbol; identity=identity/role/brand; scene=background/setting; style=image medium; other=remaining explicit information, including subjective attitude/atmosphere (give subtype).
Preserve negative scope, uncertainty and alternatives. "may hold" is positive/uncertain; "may not hold" negative/uncertain. Don't turn a local negation into global absence. Don't replace people by their uncertain identity as family/friends. Put identity as a value, not a second participant. Explicit property assignments are not actions by the property owner.
Use an empty subtype unless category=other. The literal words "none" or "null" are not substitutes for empty strings. Put may/not in their polarity/modality fields, rather than adding them to the predicate. The real participant, not the phrase "image shows", is the subject of counts and properties.
If TARGET contains multiple assertions, separate them; don't drop the action to keep just its position. Include no commentary.
EXAMPLES (unrelated source data; do not copy their facts): ''' + json.dumps(examples, ensure_ascii=False, separators=(',', ':')) + '\nSOURCE (data): ' + json.dumps(caption, ensure_ascii=False) + '\nFOCUS (data): ' + json.dumps(source_spans(caption)[focus_index], ensure_ascii=False) + '\nTARGET (data): ' + json.dumps(claim, ensure_ascii=False)


def review_schema():
    item = obj({"fact_index": {"type": "integer", "minimum": 0}, "reason": STRING})
    omission = obj({"quote": STRING, "reason": STRING})
    return obj({"unsupported": {"type": "array", "items": item},
                "misrepresented": {"type": "array", "items": item},
                "missing": {"type": "array", "items": omission}})


def review_prompt(caption, facts):
    visible = [{k: f[k] for k in ("category", "subject", "predicate", "object", "scope", "polarity", "modality", "quote")} for f in facts]
    return '''Audit the RECORDS against SOURCE as text, not against a picture or world knowledge. Return {"unsupported":[{"fact_index":0,"reason":"..."}],"misrepresented":[{"fact_index":0,"reason":"..."}],"missing":[{"quote":"exact source passage","reason":"specific missing information"}]} with empty lists where appropriate.
Check both directions: each record must preserve an assertion of SOURCE; all explicit source information must be represented. Inspect numbers, colors/material/size, actor/target direction, spatial prepositions, negation scope, uncertainty, alternatives, background, text, style and subjective assertions. A source quote attached to a wrong record does not make the record correct. A property hidden in a subject name does not replace its explicit property record. Never infer missing information from outside knowledge.
unsupported: added or reversed claims. misrepresented: wrong category, scope, certainty, object role/assignment or meaning. missing: explicit source information absent from records, even if related nouns appear. Do not report implicit entity existence as missing when already entailed by a definite positive record. Do not demand that uncertain or negated claims assert existence. A missing detail is not fixed by labeling it uncertain. This is an audit; do not output replacement facts.
Example: SOURCE="Four striped cushions lie on a velvet sofa." and one spatial record {subject:"Four striped cushions",predicate:"lie on",object:"a velvet sofa"} still lacks three separate facts: the cushions' count, their striped property, and the sofa's velvet material. Report these omissions even though those words occur in entity names.
SOURCE (data): ''' + json.dumps(caption, ensure_ascii=False) + '\nRECORDS (zero-based data): ' + json.dumps(visible, ensure_ascii=False)


def decode_review(text, caption, facts):
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"unsupported", "misrepresented", "missing"}:
        raise ValueError("AUDIT_SCHEMA_REVIEW")
    for name in ('unsupported', 'misrepresented', 'missing'):
        if not isinstance(data[name], list):
            raise ValueError("AUDIT_SCHEMA_REVIEW")
        for row in data[name]:
            fields = {'quote', 'reason'} if name == 'missing' else {'fact_index', 'reason'}
            if not isinstance(row, dict) or set(row) != fields or not isinstance(row['reason'], str) or not row['reason'].strip():
                raise ValueError("AUDIT_ITEM_REVIEW")
            if name == 'missing':
                if not isinstance(row['quote'], str) or not row['quote'].strip() or row['quote'] not in caption:
                    raise ValueError("AUDIT_EVIDENCE_REVIEW")
            elif type(row['fact_index']) is not int or not 0 <= row['fact_index'] < len(facts):
                raise ValueError("AUDIT_REFERENCE_REVIEW")
    return data


class ReviewedExtractor:
    def __init__(self, backend, store, nlp=None, max_new_tokens=1536, seed=42):
        self.backend, self.store, self.nlp, self.seed = backend, store, nlp, seed
        self.settings = dict(temperature=0, max_new_tokens=max_new_tokens)

    def call(self, stage, prompt, schema):
        request = dict(kind='reviewed_'+stage, prompt=prompt, schema=schema,
                       settings=self.settings, seed=self.seed, history=[])
        raw = self.store.cached(request, lambda: self.backend.generate_structured(
            prompt, self.settings, self.seed, schema))
        if (raw.get('history_messages', 0) or raw.get('hit_limit') or
                raw.get('stop_reason') in {'repeated_fact', 'eos_incomplete', 'token_budget'}):
            raise ValueError('INCOMPLETE_OR_HISTORY_REVIEW: '+stage)
        return raw

    def extract(self, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('Nonempty source required')
        from .syntax_audit import audit as syntax_audit
        started = time.monotonic()
        record = dict(caption=caption, caption_hash=fingerprint(caption), source_units=[],
                      facts=[], pending_queries=[], errors=[], calls=[],
                      status='REVIEW_REQUIRED', eligible_for_formal_scoring=False,
                      semantic_acceptance='PENDING_INDEPENDENT_REVIEW')
        for index, focus in enumerate(source_spans(caption)):
            unit = dict(index=index, focus=focus, claims=[])
            record['source_units'].append(unit)
            try:
                raw = self.call('decompose', decomposition_prompt(caption, focus), decomposition_schema(focus))
                record['calls'].append(dict(stage='decompose', source_index=index, raw=raw))
                claims, pending = decode_decomposition(raw['text'], caption, focus)
                unit['claims'] = claims
                for item in pending:
                    record['pending_queries'].append(dict(item, source_index=index,
                        candidate_id=None, extraction_status='unresolved', eligible_for_scoring=False))
            except (ValueError, TypeError, KeyError) as exc:
                record['errors'].append(dict(stage='decompose', source_index=index, error=str(exc)))
                continue
            for claim_index, claim in enumerate(claims):
                try:
                    raw = self.call('structure', structure_prompt(caption, index, claim), structure_schema())
                    record['calls'].append(dict(stage='structure', source_index=index, claim_index=claim_index, raw=raw))
                    facts = decode_structure(raw['text'], caption, index)
                    if any(f['quote'] != focus for f in facts):
                        raise ValueError('STRUCTURE_FOCUS_REVIEW')


                    for f in facts:
                        f['entities'] = [dict(e, id=f's{index}c{claim_index}:'+e['id']) for e in f['entities']]
                    record['facts'].extend(facts)
                except (ValueError, TypeError, KeyError) as exc:
                    record['errors'].append(dict(stage='structure', source_index=index, claim_index=claim_index, error=str(exc)))
        try:
            raw = self.call('audit', review_prompt(caption, record['facts']), review_schema())
            record['calls'].append(dict(stage='audit', raw=raw))
            record['semantic_audit'] = decode_review(raw['text'], caption, record['facts'])
        except (ValueError, TypeError, KeyError) as exc:
            record['errors'].append(dict(stage='audit', error=str(exc)))
        record['syntax_warnings'] = syntax_audit(caption, record['facts'], self.nlp) if self.nlp else []
        record['source_warnings'] = semantic_warnings(record['facts'])
        record['syntax_audit_available'] = self.nlp is not None

        if not (record['errors'] or record['pending_queries'] or record['syntax_warnings'] or record['source_warnings'] or
                any(record.get('semantic_audit', {'missing':['audit_unavailable']}).values())):
            record['status'] = 'MACHINE_CHECKS_PASSED_REVIEW_PENDING'
        record['wall_seconds'] = time.monotonic()-started
        record['model_call_count'] = len(record['calls'])
        record['output_tokens'] = sum(c['raw'].get('tokens', 0) for c in record['calls'])
        record['input_tokens'] = sum(c['raw'].get('prompt_tokens', 0) for c in record['calls'])
        self.store.event(dict(event='reviewed_extraction', caption_hash=record['caption_hash'],
                              status=record['status'], errors=record['errors']))
        return record
