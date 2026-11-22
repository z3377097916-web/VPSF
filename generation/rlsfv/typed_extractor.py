"""Development extractor: inspect all eleven categories directly in the source.

Category schedules do not depend on a preceding model's incomplete claim list.
No image access, cross-request history, semantic repair, or score changes.
"""
import json
import re
import time

from .config import CATEGORIES
from .facts import parse_facts
from .io import fingerprint
from .reviewed_extractor import obj, STRING, STRINGS, validate_pending

GROUPS = tuple((category,) for category in CATEGORIES)
assert set(sum(GROUPS, ())) == set(CATEGORIES)
STRUCTURAL = {"exists", "is", "count", "color", "material", "size", "identity",
              "text", "style", "setting", "weather", "atmosphere", "time", "mood"}
NONE = {"object", "action"}
RELATIONAL = {"spatial", "relation"}
DEFINITIONS = {
    "object": "Explicit standalone presence or absence ONLY. Entities in positive relations are handled downstream; do not infer absence from a negative relation. predicate=exists, object empty.",
    "count": "Every explicitly stated quantity, including subgroup quantities, bounds (at least), approximations, and nonnumeric quantities. One record per counted group. predicate=count, object=complete quantity phrase. Do not mistake printed digits for a quantity. Preserve subgroup restrictions in scope.",
    "attribute": "Every explicitly stated color, size, material, shape or other property. Separate properties; properties inside noun phrases must also be extracted. Keep identifying adjectives in subjects when needed to distinguish participants. Do not infer properties from common knowledge.",
    "action": "Intransitive action or posture. object empty. Separate an explicit posture/activity from its spatial position. Do not put a transitive action here: it belongs in relation.",
    "spatial": "Spatial location, direction or path between participants. object is the second participant. Include the full preposition phrase; do not drop against, left of, in front of, etc. Attribute ownership is not a spatial relation.",
    "relation": "Interactions and structural/part-whole relations. subject is the actor/part, object is the acted-on entity/whole. Preserve additional roles in scope, with their source entities in extra_entities. For giving/passing an item to someone, keep actor, item AND recipient; never invent an event actor. Wearing/holding are interactions, not attribute records.",
    "identity": "Explicit identities, roles, brands or named places. subject is the described entity; object is its identity. Preserve alternatives and may/appear. Ordinary nouns alone do not justify extra identity claims.",
    "scene": "Explicit scene/background setting, weather, lighting, and time. Separate independently stated details. Do not infer a time from lighting. Use the source's scene/image/place subject, not an invented physical entity.",
    "style": "Explicit image medium and style, such as photograph, painting, monochrome. Do not infer photographic style merely because the source says image. Separate distinct medium and style statements where both are explicit.",
    "text_ocr": "Explicit visible writing or symbols. object is the exact quoted/literal content, including digits and NO/not inside it. Those characters are not scene negation or object quantities. Do not invent unread text.",
    "other": "Explicit remaining claims, including subjective atmosphere/emotion. Preserve them as assertions made by the text; this is not image truth verification. predicate is a concrete subtype/relation; never use other for extraction failure. Extract only material outside the other ten categories.",
}


def schema(categories):
    if len(categories) != 1:
        raise ValueError('One source category per request')
    category = categories[0]
    fields = {"subject": STRING}
    if category in {"action", "relation", "spatial", "other"}:
        fields['predicate'] = STRING
    if category not in NONE:
        fields['object'] = STRING
    fields.update({"scope": STRING,
               "polarity": {"type": "string", "enum": ["positive", "negative"]},
               "modality": {"type": "string", "enum": ["definite", "uncertain"]},
               "ambiguous": {"type": "boolean"}})
    if category in RELATIONAL:
        fields['extra_entities'] = STRINGS
    return obj({'facts': {'type': 'array', 'items': obj(fields)}})


def prompt(caption, categories):
    category = categories[0]
    examples = {
        'count': 'Only written quantity words/digits qualify. "A boat floats" gives NO count record. Never write one/One merely because a/an appears. "Seven boats float" gives subject="boats", object="Seven". "At least five boxes" gives subject="boxes", object="At least five".',
        'attribute': '"A tall purple vase rests on a glass shelf" has three properties: vase/tall, vase/purple, shelf/glass. Do not omit properties simply because they occur inside a noun phrase.',
        'object': '"A kite is visible" gives subject="kite". "No horse is behind a fence" is a negative spatial relation, NOT a global absence: return no object record for it.',
        'action': '"Four travelers stand beside a kiosk" has subject="travelers", predicate="stand"; the location is excluded from this action. "A girl carries a basket" has a target and belongs in relation, not action.',
        'spatial': '"A van is behind a bus" gives van/behind/bus. Preserve full direction. "A girl holds a basket" belongs in relation and gives no spatial record.',
        'relation': '"A girl may hand a flower to a worker" gives subject="girl", predicate="hand", object="flower", scope="to a worker", extra_entities=["worker"], modality="uncertain". The actor and item never go in extra_entities. "A horse is beside a fence" is spatial, not relation.',
        'identity': '"They appear to be colleagues" gives an uncertain identity, not definite colleagues. "A woman is standing" gives NO identity record; a common noun does not require guessing a specific name.',
        'scene': 'Extract every explicit setting/weather/time detail, not locations between individual objects. An explicitly sunny day and an explicitly outdoor setting are different details.',
        'style': '"A monochrome drawing" explicitly states monochrome style and drawing medium. The word image alone gives NO style record.',
        'text_ocr': 'Only transcribed visible writing/symbols. Copy its literal contents, not their interpretation. If no writing is transcribed, return an empty facts array.',
        'other': 'Exclude existence, properties, counts, spatial positions, actions, interactions, identity, setting, time, weather, image style, and visible writing. "A purple vase stands beside a window" gives NO other record. "The atmosphere feels joyful" gives atmosphere/feels/joyful (a source claim, not verified truth).',
    }
    fields = ', '.join(schema(categories)['properties']['facts']['items']['properties'])
    return ('Extract ONLY '+category+' information from SOURCE. Read it as text; do not imagine an image. '
        +DEFINITIONS[category]+'\n'+examples[category]+'\n'
        'Return {"facts":[...]} with fields: '+fields+'. Use an empty facts array if this category is absent. '
        'Copy subject, object, predicate and scope words from the source; preserve word forms. '
        'Resolve a clear pronoun using this source. Preserve distinguishing qualifiers and subgroup restrictions. '
        'scope is an exact additional restriction, or empty. polarity is positive/negative; modality is definite/uncertain. '
        'Preserve may/might/appear and local negation, without changing unrelated claims. Keep alternatives together. '
        'ambiguous=true ONLY if a pronoun/participant truly cannot be resolved from the source. '
        'Ordinary unnamed people, plural groups, may, and absent optional properties are NOT unresolved references. '
        'Do not add missing information. Read every sentence including the last. JSON only.\nSOURCE: '
        +json.dumps(caption,ensure_ascii=False))


def source_copy(phrase, source, *, empty=False):
    "Accept exact source spans and casing differences; never lemma/synonym repair."
    if not isinstance(phrase, str) or (not phrase and not empty):
        raise ValueError('SOURCE_COPY_TYPE_OR_EMPTY')
    if not phrase:
        return ''
    pattern = (r'(?<!\w)' if re.match(r'\w', phrase[0]) else '') + re.escape(phrase)
    pattern += r'(?!\w)' if re.match(r'\w', phrase[-1]) else ''
    if re.search(pattern, source):
        return phrase
    matches = {m.group() for m in re.finditer(pattern, source, re.IGNORECASE)}
    if len(matches) != 1:
        raise ValueError('SOURCE_COPY_REVIEW: ' + phrase)
    return matches.pop()


def decode(text, caption, categories):
    raw = json.loads(text)
    if len(categories) != 1 or not isinstance(raw, dict) or set(raw) != {'facts'}:
        raise ValueError('TYPED_ROOT_REVIEW')
    fields = set(schema(categories)['properties']['facts']['items']['properties'])
    facts, origins, pending = [], [], []
    for category in categories:
        if not isinstance(raw['facts'], list):
            raise ValueError('TYPED_ARRAY_REVIEW')
        for index, wire in enumerate(raw['facts']):
            row = wire
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError('TYPED_FIELDS_REVIEW')
            if type(row['ambiguous']) is not bool or not isinstance(row.get('extra_entities', []), list):
                raise ValueError('TYPED_TYPES_REVIEW')
            fixed = {'object':'exists','count':'count','attribute':'is','identity':'identity',
                     'scene':'is','style':'style','text_ocr':'text'}
            row = dict(row, quote=caption, predicate=row.get('predicate',fixed.get(category)),
                       object=row.get('object',''),extra_entities=row.get('extra_entities',[]))
            quote = caption
            subject = source_copy(row['subject'], caption)
            value = source_copy(row['object'], caption, empty=category in NONE)
            scope = source_copy(row['scope'], caption, empty=True)
            predicate = row['predicate']
            if predicate not in STRUCTURAL:
                predicate = source_copy(predicate, caption)
            if category in NONE and value:
                raise ValueError('TYPED_UNARY_ROLE_REVIEW')
            if category == 'object' and predicate != 'exists':
                raise ValueError('TYPED_EXISTENCE_REVIEW')
            if category == 'count' and predicate != 'count':
                raise ValueError('TYPED_COUNT_REVIEW')
            extra = [source_copy(a, scope) for a in row['extra_entities']]
            if extra and category not in RELATIONAL:
                raise ValueError('TYPED_EXTRA_ROLE_REVIEW')
            physical = category in {'object', 'attribute', 'count', 'action', 'relation', 'spatial', 'identity', 'text_ocr'}
            anchors = ([subject] if physical else []) + ([value] if category in RELATIONAL else []) + extra
            anchors = list(dict.fromkeys(anchors))
            fact = dict(category=category, subject=subject, predicate=predicate, object=value,
                        scope=scope, polarity=row['polarity'], modality=row['modality'],
                        subtype=predicate if category == 'other' else '', quote=quote,
                        entities=[dict(id=fingerprint(a)[:16], anchor=a, qualifiers=[]) for a in anchors],
                        ambiguous=row['ambiguous'])
            facts.append(fact)
            origins.append(dict(category=category, index=index, raw=wire))
            if fact['ambiguous']:
                facts.pop()
                pending.append(dict(quote=caption, issue='Resolve the participant reference in this source claim: '+
                                    ' | '.join([subject,predicate,value,scope]),
                                    anchors=[a for a in anchors if a.casefold() not in {'he','she','it','they','them','this','that','these','those'}]))
    return parse_facts(json.dumps({'facts': facts}), caption), pending, origins


class TypedExtractor:
    def __init__(self, backend, store, max_new_tokens=2048, seed=42):
        self.backend, self.store, self.seed = backend, store, seed
        self.settings = dict(temperature=0, max_new_tokens=max_new_tokens)

    def extract(self, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('Nonempty source required')
        start = time.monotonic()
        result = dict(caption=caption, caption_hash=fingerprint(caption), facts=[], pending_queries=[],
                      errors=[], calls=[], origins=[], eligible_for_formal_scoring=False,
                      status='REVIEW_REQUIRED', semantic_acceptance='PENDING_INDEPENDENT_REVIEW')
        for categories in GROUPS:
            request = dict(kind='typed_source_extraction', prompt=prompt(caption, categories),
                           schema=schema(categories), categories=categories,
                           settings=self.settings, seed=self.seed, history=[])
            try:
                raw = self.store.cached(request, lambda: self.backend.generate_structured(
                    request['prompt'], self.settings, self.seed, request['schema']))
                result['calls'].append(dict(categories=categories, raw=raw))
                if raw.get('history_messages') or raw.get('hit_limit') or raw.get('stop_reason') in {'eos_incomplete', 'token_budget', 'repeated_fact'}:
                    raise ValueError('INCOMPLETE_OR_HISTORY_REVIEW')
                facts, pending, origins = decode(raw['text'], caption, categories)
                result['facts'].extend(facts)
                result['origins'].extend(origins)
                result['pending_queries'].extend(dict(x, eligible_for_scoring=False) for x in pending)
            except (ValueError, KeyError, TypeError) as exc:
                result['errors'].append(dict(categories=categories, error=str(exc)))
        if not result['errors'] and not result['pending_queries'] and result['facts']:
            result['status'] = 'MACHINE_CHECKS_PASSED_REVIEW_PENDING'
        result['wall_seconds'] = time.monotonic()-start
        result['model_call_count'] = len(result['calls'])
        result['output_tokens'] = sum(x['raw'].get('tokens', 0) for x in result['calls'])
        return result
