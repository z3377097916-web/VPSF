#!/usr/bin/env python3
"""Read saved records only. Never call a model or re-extract textual facts.

Produces the *training-time feature ABI* of the supplied frozen formula weights.
Aux/pending omissions are documented in legacy_coco_v1, not silently redefined.
"""
import argparse,json,gzip,zipfile,io,sys,hashlib,shutil,tempfile
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from preference.compact import features
from preference.inputs import image_key
from preference.source_contract import raw_source,file_hash

def unwrap(d):
    return d["result"] if "result" in d and "candidates" not in d else d

def records(path,member=None):
    if member:
        with zipfile.ZipFile(path) as z,z.open(member) as stream,gzip.GzipFile(fileobj=stream) as g:
            for line in g:yield unwrap(json.loads(line))
    elif path.is_dir():
        for p in sorted(path.rglob('result.json')):yield unwrap(json.loads(p.read_text()))
    else:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as f:
            for line in f:yield unwrap(json.loads(line))

def status(f):
    s=[x.get('status') for x in f.get('checks',[])]
    if 'refuted' in s:return 'unsupported'
    if s and all(x=='supported' for x in s):return 'supported'
    return 'uncertain'

def extra(d):
    facts=d['facts'];iid=image_key(d['record']['id']);out={};cons=[];cons2=[]
    eligible={r['id']:r['eligible'] for r in d['ranking']['rows']};objs={}
    for c in d['candidates']:
        counts={cat:dict(supported=0,uncertain=0,unsupported=0) for cat in ('object','relation','aux')};ds=[];o={}
        for k in c['claim_ids']:
            f=facts[k]
            if f['modality']!='definite' or f['polarity']!='positive':continue
            cat=f['category'];st=status(f)
            if cat in counts:counts[cat][st]+=1
            if cat=='object':
                ss=[x.get('status') for x in f['checks']]
                q=0 if 'refuted' in ss else (1 if ss.count('supported')==2 else .5 if ss.count('supported')==1 and ss.count('unclear')==1 else 0)
                ds.append(1-q);name=' '.join(str(f.get('subject') or '').lower().split())
                if name and (name not in o or o[name]!='supported' and st=='supported'):o[name]=st
        row={}
        for cat,v in counts.items():row.update({cat+'_n':sum(v.values()),cat+'_sup':v['supported'],cat+'_unc':v['uncertain'],cat+'_uns':v['unsupported']})
        row['Do']=sum(ds)/len(ds) if ds else 0
        out[iid+'|'+c['id']]=row
        if eligible[c['id']]:objs[c['id']]=o
    holders=defaultdict(dict)
    for cid,o in objs.items():
        for n,s in o.items():holders[n][cid]=s
    for cid,o in objs.items():
        n=len(o);den=max(1,n);multi=len(objs)>1
        both=sum(sum(v=='supported' for v in holders[name].values())>=2 for name in o) if multi else 0
        freq=sum(len(holders[name]) for name in o)/den if multi else 0
        ext_cov=no_ext=0
        if multi:
            for name,st in o.items():
                sup=any(k!=cid and v=='supported' for k,v in holders[name].items())
                ext_cov+=int(st=='supported' and sup);no_ext+=int(not sup)
        cons.append(dict(image_id=iid,cand=cid,obj_n=n,cons_sup_frac=both/den,freq_mean=freq))
        cons2.append(dict(image_id=iid,cand=cid,ext_sup_cov_frac=ext_cov/den,no_ext_frac=no_ext/den))
    return out,cons,cons2

def compact_record(d):
    cs=features(d);rows={r['id']:r for r in d['ranking']['rows']}
    for c in cs:
        r=rows[c['id']];c.update(bf=r['scores']['Bf'],ranks=r['rank_points'],coverage=r['coverage'])
        # G uses valid/effective checks; the frozen R_A/R_B input below retains
        # its historical raw-status definition. Do not conflate these fields.
        c['g_double_errors']=list(c['double_errors'])
        c['double_errors']=sorted(k for k in c['claim_ids'] if [x.get('status') for x in d['facts'][k]['checks']]==['refuted','refuted'])
        c['unsupported_primary_fraction']=len(set(c['primary_claims'])-set(c['double_supported_claims']))/max(1,len(c['primary_claims']))
    iid=image_key(d['record']['id'])
    return dict(image_id=iid,source_record=dict(d['record'],id=iid),image_sha256=d.get('image_sha256'),
        source_sha256=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),round=d.get('round'),
        candidates=cs,selections=d['ranking']['selections'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--member');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--diagnostic',action='store_true',help='Historical replay only; output is not trainable')
    a=ap.parse_args();expected=None;identity=None;source=None;runs=set()
    if a.input.is_dir():


        run=json.loads((a.input/'run.json').read_text())
        identity={k:v for k,v in run.items() if k!='records'}
        state=json.loads((a.input/'status.json').read_text())
        if not state.get('complete'):raise ValueError('Generation is incomplete; failed images retained, not omitted')
        expected={image_key(r['id']) for r in run['records']}
        if len(expected)!=len(run['records']):raise ValueError('Duplicate image id after normalization')
    if a.output.exists():raise FileExistsError(a.output)
    ims=[];af={};cs=[];cs2=[];seen=set()
    for d in records(a.input,a.member):
        if not a.diagnostic:
            current,run_hash=raw_source(d,identity)
            if source is not None and current!=source:raise ValueError('Mixed source models/prompts')
            source=current;runs.add(run_hash)
        im=compact_record(d)
        if not a.diagnostic:im.update(source_contract=source,source_run_hash=run_hash)
        if im['image_id'] in seen:raise ValueError('Duplicate image id')
        seen.add(im['image_id']);ims.append(im);f,c,c2=extra(d);af.update(f);cs.extend(c);cs2.extend(c2)
        if len(ims)%50==0:print('Read',len(ims),'images',flush=True)
    if not ims:raise ValueError('No complete records found')
    if expected is not None and seen!=expected:raise ValueError('Saved result IDs differ from complete run manifest')


    a.output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.tmp-',dir=a.output.parent))
    try:
        for name,val in [('candidate_features.json',ims),('all_fact_features.json',af),('consensus_features.json',cs),('consensus_features2.json',cs2)]:
            (staging/name).write_text(json.dumps(val,ensure_ascii=False),encoding='utf8')
        (staging/'feature_contract.json').write_text(json.dumps(dict(schema='legacy_coco_v1',images=len(ims),
            usage='diagnostic' if a.diagnostic else 'training',source=source,source_runs=sorted(runs),
            files={p.name:file_hash(p) for p in sorted(staging.glob('*.json'))},
            g_fallback_schema='effective_double_refutations_v1',
            note='Fitted ABI: relation excludes spatial here; aux and pending model dimensions were constant zero. Primary scoring still includes spatial. Raw double_errors are frozen formula inputs; g_double_errors are valid effective G checks.'),indent=2),encoding='utf8')
        if a.output.exists():raise FileExistsError(a.output)
        staging.rename(a.output)
    finally:
        if staging.exists():shutil.rmtree(staging)
if __name__=='__main__':main()
