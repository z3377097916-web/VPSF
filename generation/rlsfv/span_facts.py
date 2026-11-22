"Experimental flat, source-indexed facts, without LLM entity-ID bookkeeping."
import json
from .anchored import spans
from .config import CATEGORIES
from .structured import source_spans, validate_extraction

STRUCTURAL={'exists','is','count','color','material','size','identity','text','style',
            'setting','weather','atmosphere'}

def schema():
    def obj(p):
        return dict(type='object',properties=p,required=list(p),additionalProperties=False)
    string=dict(type='string')
    def choice(values):return dict(type='string',enum=list(values))
    return obj({'facts':dict(type='array',items=obj({
        'subject':string,'predicate':string,'object':string,
        'object_role':choice(['entity','value','none']),
        'polarity':choice(['positive','negative']),
        'modality':choice(['definite','uncertain']),
        'category':choice(CATEGORIES),'scope':string,'subtype':string,
        'ambiguous':dict(type='boolean'),'evidence':dict(type='integer',minimum=0)}))})

def prompt(caption):
    def f(s,p,o,role,c,n='positive',m='definite',ev=0):
        return dict(subject=s,predicate=p,object=o,object_role=role,polarity=n,
                    modality=m,category=c,scope='',subtype='',ambiguous=False,evidence=ev)
    examples=[
        {'sentences':['Two green bottles are on a metal tray.'], 'facts':[
            f('bottles','count','Two','value','count'),
            f('bottles','color','green','value','attribute'),
            f('tray','material','metal','value','attribute'),
            f('bottles','on','tray','entity','spatial')]},
        {'sentences':['A child might touch a ball.','No cat is under the desk.'], 'facts':[
            f('child','touch','ball','entity','relation',m='uncertain'),
            f('cat','under','desk','entity','spatial',n='negative',ev=1)]}]
    return '''Extract every distinct claim explicitly expressed in the numbered SOURCE SENTENCES. Return only {"facts":[...]}.
Read the text, do not imagine its picture. Preserve incorrect and subjective source claims too. Do not add explanations.
Each record uses normal field names:
subject: exact source referring phrase, retaining words needed to distinguish same-kind participants; resolve clear pronouns to their source antecedent. A plural group is one subject, not ambiguous merely because it is plural.
predicate: copy the relation/action phrase, including required prepositions, or use a structural label exists/is/count/color/material/size/identity/text/style/setting/weather/atmosphere.
object: exact source target/value phrase, or empty. object_role: entity for a second participant, value for an attribute/number/identity/writing, none for no object. Do not output entity numbers or entity tables.
polarity: positive means affirmed, negative means denied. modality: definite means asserted, uncertain means may/might/appear/seem. These fields are independent.
category: object (existence), attribute (one explicit property), count (quantity), spatial (position/path), action (intransitive activity/posture), relation (interaction/part-whole), text_ocr (writing/symbol), identity (identity/role/brand), scene (setting/background), style (image medium), other (remaining explicit information).
scope: exact explicit restriction if needed, otherwise empty. subtype: concrete kind for other, otherwise empty. ambiguous: true only when the participant reference or reading is unresolved. evidence: the zero-based source sentence number. The program attaches the original sentence and participant IDs; do not copy sentences or generate IDs yourself.
Preserve quantities (including at least/approximately), negation, alternatives, uncertainty and direction. "may touch" is positive/uncertain. "does not touch" is negative/definite. "may not touch" is negative/uncertain. A denied spatial relation must not become global nonexistence. Do not infer existence from uncertain or negated statements. An alternative such as "family or friends" is one value, never two certain identities. Do not invent causal links, intentions or part-whole relations.
Each independently stated color/material/size, count, activity and position gets its own record; don't hide them in another record. "standing beside" has an activity and a position. If existence is already entailed by a definite positive claim, do not repeat an inferred object record: the existing downstream program handles entailed existence. Explicit standalone existence statements still need object records. Qualifiers that distinguish two participants remain in their referring phrases as well as any explicitly stated property records.
Examples (separate data, do not copy their facts into the actual output):
'''+json.dumps(examples,ensure_ascii=False,separators=(',',':'))+'''
Now process all numbered SOURCE SENTENCES, with full context available for references. Check for omitted properties, counts and relations before finishing. Return JSON only.
SOURCE SENTENCES:
'''+json.dumps(dict(enumerate(source_spans(caption))),ensure_ascii=False)

def decode(text,caption):
    data=json.loads(text)
    if not isinstance(data,dict) or set(data)!={'facts'} or not isinstance(data['facts'],list):
        raise ValueError('SPAN_ROOT_REVIEW')
    fields=set(schema()['properties']['facts']['items']['properties'])
    sentences=source_spans(caption);phrases=set(spans(caption));ids={};facts=[]
    for row in data['facts']:
        if not isinstance(row,dict) or set(row)!=fields:raise ValueError('SPAN_FIELDS_REVIEW')
        if any(not isinstance(row[k],str) for k in fields-{'ambiguous','evidence'}):raise ValueError('SPAN_TYPES_REVIEW')
        if type(row['ambiguous']) is not bool or type(row['evidence']) is not int:
            raise ValueError('SPAN_TYPES_REVIEW')
        ev=row['evidence']
        if not 0<=ev<len(sentences):raise ValueError('SPAN_EVIDENCE_REVIEW')
        local=set(spans(sentences[ev]));role=row['object_role']
        if role not in {'entity','value','none'}:raise ValueError('SPAN_ROLE_REVIEW')
        if (role=='none') != (row['object']==''):raise ValueError('SPAN_ROLE_REVIEW')
        expected_roles = {'object': 'none', 'attribute': 'value', 'count': 'value',
                          'identity': 'value', 'text_ocr': 'value', 'style': 'value',
                          'spatial': 'entity', 'relation': 'entity', 'action': 'none'}
        if row['category'] in expected_roles and role != expected_roles[row['category']]:
            raise ValueError('SPAN_CATEGORY_ROLE_REVIEW')
        if row['subject'] not in phrases:raise ValueError('SPAN_SUBJECT_COPY_REVIEW')

        valid_objects=phrases if role=='entity' else local if role=='value' else {''}
        if row['object'] not in valid_objects:raise ValueError('SPAN_OBJECT_COPY_REVIEW')
        if row['predicate'] not in local|STRUCTURAL or row['scope'] not in local|{''}:
            raise ValueError('SPAN_PREDICATE_SCOPE_COPY_REVIEW')
        anchors=list(dict.fromkeys([row['subject']]+([row['object']] if role=='entity' else [])))
        fact={k:row[k] for k in fields-{'evidence','object_role'}}
        fact['quote']=sentences[ev]
        fact['entities']=[dict(id=ids.setdefault(a,'e'+str(len(ids))),anchor=a,qualifiers=[]) for a in anchors]
        facts.append(fact)
    return validate_extraction(json.dumps({'facts':facts},ensure_ascii=False),caption)
