"""Extraction-only format constraints. These do NOT establish semantic correctness.

No response repair, partial JSON acceptance, automatic fact deletion, or semantic
category changes. Optional sentence calls retain the complete caption as context.
"""
import json
import re
from .config import CATEGORIES
from .facts import parse_facts, norm


def source_spans(caption):
    """Contiguous source units; keep every character, including unpunctuated tails.

Conservative punctuation boundaries are scheduling units, not a linguistic parser.
The complete caption is still passed to every request for reference resolution.
"""
    boundaries = [0]
    for match in re.finditer(r'[.!?](?:["\u201d\u2019])?\s+(?=[A-Z])', caption):
        boundaries.append(match.end())
    boundaries.append(len(caption))
    return [caption[a:b] for a, b in zip(boundaries, boundaries[1:]) if b > a]


def extraction_schema(quotes, compact=False):
    string = {"type": "string"}
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    entity = obj({"id": string, "anchor": string,
                  "qualifiers": {"type": "array", "items": string}})
    fact = obj({"quote": {"type": "string", "enum": quotes},
                "category": {"type": "string", "enum": list(CATEGORIES)},
                "subject": string, "predicate": string, "object": string,
                "scope": string,
                "polarity": {"type": "string", "enum": ["positive", "negative"]},
                "modality": {"type": "string", "enum": ["definite", "uncertain"]},
                "subtype": string, "entities": {"type": "array", "items": entity},
                "ambiguous": {"type": "boolean"}})
    if compact:


        del fact['properties']['quote']
        fact['required'].remove('quote')
        del entity['properties']['id']
        entity['required'].remove('id')

    return obj({"facts": {"type": "array", "items": fact}})


def extraction_prompt(caption, focus, compact=False):
    prompt = '''Extract every distinct atomic fact stated in the FOCUS passage. The full caption is supplied only to resolve references. Do not extract facts stated only outside FOCUS. Do not describe an image or add plausible details. Return {"facts":[...]} only.
Each fact: quote (copy FOCUS exactly), category, subject, predicate, object, scope, polarity, modality, subtype, entities, ambiguous. Entities contain id, anchor, qualifiers (a string list). Use empty strings/lists for unused fields. Preserve direction, counts, negation and uncertainty. "may", "might", "appear" and alternatives are not definite claims. Do not convert a negated claim into positive existence. Preserve different entity identities; mark unresolved identity ambiguous. No repeated facts.
Categories: object=existence; attribute=property; count=quantity; spatial=position; action=action/posture; relation=interaction/part-whole; text_ocr=visible text; identity=specific identity; scene=background; style=medium/style; other=remaining explicit information (specify subtype).
FULL CAPTION (data): ''' + json.dumps(caption, ensure_ascii=False) + '\nFOCUS (data): ' + json.dumps(focus, ensure_ascii=False)
    if compact:
        prompt = prompt.replace('quote (copy FOCUS exactly), ', '').replace('Entities contain id, anchor, qualifiers', 'Entities contain anchor, qualifiers')
    return prompt


def expand_wire(text, focus):
    """Attach scheduled source and local IDs without inferring semantic fields.

The attached quote locates the requested passage; it is NOT proof of entailment.
"""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'facts'} or not isinstance(value['facts'], list):
        raise ValueError('Invalid wire root')
    anchors = {}
    for f in value['facts']:
        if not isinstance(f, dict) or 'quote' in f or not isinstance(f.get('entities'), list):
            raise ValueError('Invalid wire fact')
        f['quote'] = focus
        for e in f['entities']:
            if not isinstance(e, dict) or set(e) != {'anchor', 'qualifiers'}:
                raise ValueError('Invalid wire entity')
            if not isinstance(e['anchor'], str) or not isinstance(e['qualifiers'], list) or any(not isinstance(x, str) for x in e['qualifiers']):
                raise ValueError('Invalid wire anchor/qualifiers')
            key = json.dumps([e['anchor'], e['qualifiers']], ensure_ascii=False)
            e['id'] = anchors.setdefault(key, 'e' + str(len(anchors)+1))
    return json.dumps(value, ensure_ascii=False)


def complete_json(text):
    try:
        value = json.loads(text)
        return isinstance(value, dict) and set(value) == {"facts"} and isinstance(value['facts'], list)
    except (ValueError, TypeError):
        return False


def schema_progress(text, schema):
    """Complete-root detection for experimental schemas; never semantic approval."""
    expected = set(schema.get('properties', {}))
    if expected == {'facts'}:
        return generation_progress(text)
    try:
        value = json.loads(text)
        complete = isinstance(value, dict) and set(value) == expected
    except (ValueError, TypeError):
        complete = False
    return {'complete': complete, 'repeated': False}


def generation_progress(text):
    """Inspect completed array members ONLY to stop a loop; never salvage output."""
    match = re.match(r'^\s*\{\s*"facts"\s*:\s*\[', text)
    if not match:
        return {"complete": complete_json(text), "completed_facts": 0, "repeated": False}
    pos, seen, count = match.end(), set(), 0
    decoder = json.JSONDecoder()
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] == ']':
            break
        try:
            fact, end = decoder.raw_decode(text, pos)
        except ValueError:
            break
        if not isinstance(fact, dict):
            break

        payload = {k: v for k, v in fact.items() if k not in {'quote', 'entities'}}
        if 'entities' in fact:
            payload['entities'] = [{k: v for k, v in e.items() if k != 'id'}
                                   for e in fact['entities'] if isinstance(e, dict)]
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        count += 1
        if key in seen:
            return {"complete": complete_json(text), "completed_facts": count, "repeated": True}
        seen.add(key)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ',':
            break
        pos += 1
    return {"complete": complete_json(text), "completed_facts": count, "repeated": False}


def validate_extraction(text, caption, focus=None):
    facts = parse_facts(text, caption)
    if not facts and caption.strip():
        raise ValueError('EMPTY_EXTRACTION_REVIEW: nonempty caption produced no facts')
    seen = set()
    for f in facts:
        if focus is not None and f['quote'] != focus:
            raise ValueError('SOURCE_SCOPE_MISMATCH: evidence is outside scheduled source passage')

        key = tuple(norm(f[k]) for k in ('category', 'subject', 'predicate', 'object',
                                         'scope', 'polarity', 'modality', 'subtype')) + (
            json.dumps(sorted((norm(e['anchor']), sorted(map(norm, e['qualifiers'])))
                              for e in f['entities']), ensure_ascii=False), f['ambiguous'])
        if key in seen:
            raise ValueError('REPEATED_FACT_REVIEW: repeated proposition; do not silently discard')
        seen.add(key)
    return facts


def semantic_warnings(facts):
    """Conservative review flags, never corrections or claims of entailment.

A cue can apply to just one clause; hence these are review diagnostics rather
than automatically changing every fact's polarity/modality.
"""
    warnings = []
    for i, f in enumerate(facts):
        q = f['quote']


        cue_text = q
        if f['category'] == 'text_ocr' and f.get('object') and f['object'] in q:
            cue_text = q.replace(f['object'], '[OCR literal]', 1)
        if f['category'] in {'object', 'attribute', 'count', 'spatial', 'action', 'relation', 'text_ocr', 'identity'} and not f['entities']:
            warnings.append({'fact_index': i, 'code': 'MISSING_ENTITY_BINDING_REVIEW'})
        if re.search(r'\b(may|might|perhaps|possibly|appear\w*|seem\w*)\b', cue_text, re.I) and f['modality'] == 'definite':
            warnings.append({'fact_index': i, 'code': 'UNCERTAINTY_SCOPE_REVIEW'})
        if re.search(r'\b(no|not|never|without)\b', cue_text, re.I) and f['polarity'] == 'positive':
            warnings.append({'fact_index': i, 'code': 'NEGATION_SCOPE_REVIEW'})
        source_words = re.findall(r'\w+', norm(q))
        def explicit_phrase(phrase):
            words = re.findall(r'\w+', norm(phrase))
            return bool(words) and any(source_words[j:j+len(words)] == words
                                       for j in range(len(source_words)-len(words)+1))
        for e in f['entities']:
            if not explicit_phrase(e['anchor']):
                warnings.append({'fact_index': i, 'code': 'ENTITY_GROUNDING_REVIEW', 'anchor': e['anchor']})
            for qualifier in e['qualifiers']:
                if not explicit_phrase(qualifier):
                    warnings.append({'fact_index': i, 'code': 'QUALIFIER_GROUNDING_REVIEW', 'qualifier': qualifier})
    return warnings


def extract_caption(backend, store, caption, settings, seed):
    """Experimental stateless sentence extraction; review issues stop the image.

All source units are scheduled. No previous answer, model conclusion or history is
passed to another call. Repeated source facts remain available for canonical union.
"""
    from .io import fingerprint
    spans = source_spans(caption)
    if ''.join(spans) != caption:
        raise AssertionError('Source scheduling lost text')
    combined, records = [], []
    for index, focus in enumerate(spans):
        prompt = extraction_prompt(caption, focus, compact=True)
        schema = extraction_schema([focus], compact=True)
        request = {'kind': 'structured_claim_extraction', 'prompt': prompt,
                   'settings': settings, 'schema': schema, 'seed': seed,
                   'history': [], 'source_index': index}
        raw = store.cached(request, lambda: backend.generate_structured(prompt, settings, seed, schema))
        if raw.get('stop_reason') == 'repeated_fact':
            raise ValueError('REPEATED_FACT_REVIEW: stopped generation loop; no partial facts accepted')
        if raw.get('history_messages', 0) or raw.get('hit_limit'):
            raise ValueError('STRUCTURED_EXTRACTION_REVIEW: history or truncated output')
        facts = validate_extraction(expand_wire(raw['text'], focus), caption, focus)
        warnings = semantic_warnings(facts)
        store.event({'event': 'structured_extraction_review', 'caption_hash': fingerprint(caption),
                     'source_index': index, 'warnings': warnings, 'fact_count': len(facts)})
        if warnings:
            raise ValueError('SEMANTIC_REVIEW_REQUIRED: ' + json.dumps(warnings))

        for f in facts:
            f['entities'] = [dict(e, id=f's{index}:' + e['id']) for e in f['entities']]
        combined.extend(facts)
        records.append(raw)
    return combined, records


def make_token_enforcer(tokenizer_data, schema):
    """Preserve explicit config after lm-format-enforcer 0.11.3 resets it."""
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    from lmformatenforcer.tokenenforcer import TokenEnforcer
    config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet,
        max_json_array_length=0, force_json_field_order=True, max_consecutive_whitespaces=2)
    parser = JsonSchemaParser(schema, config=config)
    enforcer = TokenEnforcer(tokenizer_data, parser)


    parser.config = config
    return enforcer
