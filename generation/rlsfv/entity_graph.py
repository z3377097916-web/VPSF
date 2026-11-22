"""Experimental compact entity/association wire format inspired by UIE/REBEL.
No new entity existence is inferred from the entity table. Only fact participants
are reconstructed; unchanged downstream facts retain polarity, modality and scope.
"""
import json
from .config import CATEGORIES
from .structured import validate_extraction
from .anchored import spans

def schema():
    def obj(fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
    string=dict(type='string')
    entity=obj({'anchor':string,'qualifiers':dict(type='array',items=string),'ambiguous':dict(type='boolean')})
    fact=obj({'c':dict(type='string',enum=list(CATEGORIES)),
        's':dict(type='integer',minimum=0),'p':string,
        'r':dict(type='integer',minimum=-1),'v':string,
        'n':dict(type='string',enum=['positive','negative']),
        'm':dict(type='string',enum=['definite','uncertain']),
        'x':string,'t':string,'a':dict(type='boolean'),'q':string})
    return obj({'entities':dict(type='array',items=entity),'facts':dict(type='array',items=fact)})

def prompt(caption):
    example={
        'entities':[{'anchor':'bottles','qualifiers':[],'ambiguous':False},{'anchor':'tray','qualifiers':[],'ambiguous':False}],
        'facts':[]}
    for c,s,p,r,v in [('object',0,'exists',-1,''),('object',1,'exists',-1,''),('count',0,'count',-1,'Two'),('attribute',0,'color',-1,'green'),('attribute',1,'material',-1,'metal'),('spatial',0,'on',1,'')]:
        example['facts'].append(dict(c=c,s=s,p=p,r=r,v=v,n='positive',m='definite',x='',t='',a=False,q='Two green bottles are on a metal tray.'))
    return '''Convert the ACTUAL TEXT into all of its atomic claims. This is text extraction, not image verification. Preserve wrong or subjective source claims; do not guess what a picture contains. Return JSON only.
Use two lists in ONE response:
entities: each distinct participant once, with exact original anchor phrase, exact qualifiers needed for identity, ambiguous boolean. Index them from 0 by list position. A plural group such as "bottles" is ONE entry, not duplicate entries for each member. Listing an entity does NOT assert that it exists: negative/possible participants may be listed. Resolve pronouns only if unambiguous; otherwise mark ambiguity. An alternative such as "field or park" stays ONE disjunctive phrase, not two certain entities.
facts: c category; s subject entity index; p predicate; r target entity index or -1; v literal value or empty; n polarity; m modality; x explicit scope or empty; t subtype or empty; a unresolved ambiguity; q exact supporting source quotation. For a relationship use r and empty v. For a literal (number/color/material/identity/text) use r=-1 and v. For intransitive action/existence use r=-1,v="". Do not repeat entities inside facts. q must contain the claim and any modal/negative qualifier; do not copy examples into the actual output.
Atomicity and label guidelines:
object: existence only, p=exists; never assert a negated/uncertain participant definitely exists.
attribute: each explicit property separately (color, material, size, condition); literal value in v. Entity qualifiers do NOT replace attribute facts.
count: explicit quantity in v, retaining at least/approximately/etc; a count hidden in an anchor does NOT replace a count fact.
spatial: relative position/path such as on, beside, left of, through. Keep the direction, s to r.
action: action or posture such as walking, standing, leaning. Split an accompanying spatial relation into its own fact when separately stated.
relation: interaction or part-whole relationship, e.g. holding a cup, wheel part of bicycle. Do not label "beside" as relation; it is spatial.
text_ocr: exact quoted writing/symbol content in v; keep numbers/case, do not correct spelling.
identity: stated specific identity, role, brand, or collective identity. Keep "family or friends" as one alternative value, retaining uncertainty.
scene: explicit setting/background/environment facts.
style: explicit image medium/style, e.g. photograph, watercolor.
other: other stated information; t must name its concrete subtype. Preserve subjective atmosphere/feelings as source claims, not image-verified truth.
Polarity and modality are INDEPENDENT: "may hold" = positive + uncertain; "does not hold" = negative + definite; "may not hold" = negative + uncertain. Negative is not a synonym for uncertain. "No cat beside desk" denies only that spatial claim, not all cats everywhere. Preserve clauses under appear/seem/may, alternatives and negation. No causal/intention/gender/material additions.
Example text (NOT actual input): Two green bottles are on a metal tray.
Example output: '''+json.dumps(example,separators=(',',':'))+'''
Before finishing, check that every explicit count, attribute, action, spatial/interaction claim and qualifier in the ACTUAL TEXT is represented; don't emit that check. Correct formatting does not excuse missing facts.
ACTUAL TEXT: '''+json.dumps(caption,ensure_ascii=False)

def decode(text,caption):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!={'entities','facts'}:raise ValueError('GRAPH_ROOT_REVIEW')
    entities=value['entities']
    if not isinstance(entities,list) or not isinstance(value['facts'],list):raise ValueError('GRAPH_LIST_REVIEW')
    source_phrases=set(spans(caption))
    signatures=set()
    for i,e in enumerate(entities):
        if not isinstance(e,dict) or set(e)!={'anchor','qualifiers','ambiguous'} or type(e['ambiguous']) is not bool:raise ValueError('GRAPH_ENTITY_REVIEW')
        if not isinstance(e['anchor'],str) or e['anchor'] not in source_phrases:raise ValueError('GRAPH_ANCHOR_REVIEW')
        if not isinstance(e['qualifiers'],list) or any(not isinstance(q,str) or q not in source_phrases for q in e['qualifiers']):raise ValueError('GRAPH_QUALIFIER_REVIEW')
        signature=json.dumps([e['anchor'],e['qualifiers']])
        if signature in signatures and not e['ambiguous']:raise ValueError('GRAPH_DUPLICATE_ENTITY_REVIEW')
        signatures.add(signature)
    facts=[]
    for row in value['facts']:
        if not isinstance(row,dict) or set(row)!={'c','s','p','r','v','n','m','x','t','a','q'}:raise ValueError('GRAPH_FACT_FIELDS_REVIEW')
        if type(row['a']) is not bool or any(not isinstance(row[k],str) for k in ['c','p','v','n','m','x','t','q']):raise ValueError('GRAPH_FACT_TYPES_REVIEW')
        s,r=row['s'],row['r']
        if type(s) is not int or type(r) is not int or not 0<=s<len(entities) or not -1<=r<len(entities):raise ValueError('GRAPH_REFERENCE_REVIEW')
        if r>=0 and row['v']!='':raise ValueError('GRAPH_TARGET_VALUE_CONFLICT')
        participants=list(dict.fromkeys([s]+([r] if r>=0 else [])))
        fact=dict(category=row['c'],subject=entities[s]['anchor'],predicate=row['p'],
            object=entities[r]['anchor'] if r>=0 else row['v'],polarity=row['n'],modality=row['m'],
            scope=row['x'],subtype=row['t'],quote=row['q'],
            ambiguous=row['a'] or any(entities[i]['ambiguous'] for i in participants),
            entities=[dict(id='e'+str(i),anchor=entities[i]['anchor'],qualifiers=entities[i]['qualifiers']) for i in participants])
        facts.append(fact)

    return validate_extraction(json.dumps({'facts':facts},ensure_ascii=False),caption)
