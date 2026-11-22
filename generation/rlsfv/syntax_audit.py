"Source-dependency cross-check inspired by caption-to-scene-graph parsers."
from collections import Counter

def audit(caption, facts, nlp):
    doc=nlp(caption)
    rows=[(i,f) for i,f in enumerate(facts) if isinstance(f,dict) and
          all(isinstance(f.get(k),str) for k in ['subject','predicate','object','category'])]


    literal_spans=[]
    for _,f in rows:
        if f['category']!='text_ocr' or not f['object'] or not f.get('quote'):continue
        quote=f['quote'];literal=f['object'];local=quote.find(literal)
        if local<0:continue
        start=caption.find(quote)
        while start>=0:
            literal_spans.append((start+local,start+local+len(literal)))
            start=caption.find(quote,start+1)
    cache={}
    def lemmas(text):
        if text not in cache:
            cache[text]={t.lemma_.casefold() for t in nlp(text) if not t.is_punct}
        return cache[text]
    nouns=Counter(t.lemma_.casefold() for t in doc if t.pos_ in {'NOUN','PROPN'})
    out=[]
    for token in doc:
        if any(a<=token.idx and token.idx+len(token.text)<=b for a,b in literal_spans):continue
        if token.head.pos_ not in {'NOUN','PROPN'}:continue
        if token.dep_=='nummod':kind='count'
        elif token.dep_=='amod' and token.pos_=='ADJ':kind='attribute'
        elif token.dep_=='compound' and token.pos_=='NOUN':kind='compound_review'
        else:continue
        owner=token.head.lemma_.casefold();modifier=token.lemma_.casefold()
        matches=[i for i,f in rows if owner in lemmas(f['subject']) and
                 modifier in lemmas(f['object']) and
                 (f['category']==kind or kind=='compound_review')]
        if not matches or nouns[owner]>1:
            out.append(dict(code='COMPOUND_MODIFIER_REVIEW' if kind=='compound_review' else
                            'DEPENDENCY_'+kind.upper()+'_COVERAGE_REVIEW',
                            source=token.text,owner=token.head.text,
                            source_offset=[token.idx,token.idx+len(token.text)],
                            owner_offset=[token.head.idx,token.head.idx+len(token.head.text)],
                            sentence=token.sent.text,matching_record_indices=matches,
                            repeated_owner_requires_identity_review=nouns[owner]>1,
                            disposition='review_only_not_a_fact_or_truth_label'))
    for verb in doc:
        if verb.pos_!='VERB':continue
        subjects=[t for t in verb.children if t.dep_=='nsubj' and t.pos_ in {'NOUN','PROPN'}]
        objects=[t for t in verb.children if t.dep_ in {'dobj','obj'} and t.pos_ in {'NOUN','PROPN'}]


        for subject in subjects:
            for obj in objects:
                if subject.lemma_==obj.lemma_:continue
                for i,f in rows:
                    if (verb.lemma_.casefold() in lemmas(f['predicate']) and
                        obj.lemma_.casefold() in lemmas(f['subject']) and
                        subject.lemma_.casefold() in lemmas(f['object'])):
                        out.append(dict(code='DEPENDENCY_REVERSED_ROLES_REVIEW',fact_index=i,
                            source_subject=subject.text,source_predicate=verb.text,
                            source_object=obj.text,sentence=verb.sent.text,
                            disposition='review_only_not_automatic_correction'))
    return out
