"""Co-primary outcomes and every-metric gates, without averaging away regression."""
import random
from .io import read_json,atomic_json

PRIMARY_ENDPOINTS=('generative.Cover','generative.CHAIR','generative.Hal',
                   'relation.spatial_and_relation.precision','relation.spatial_and_relation.recall')

def holm(pvalues,alpha=0.05):
    """Holm step-down over the pre-registered primary family. Missing p -> not tested."""
    tested=sorted(((p,k) for k,p in pvalues.items() if p is not None))
    m=len(tested); out={k:{'p':p,'p_holm':None,'reject_at_0.05':None} for k,p in pvalues.items()}
    running=0.0; stop=False
    for i,(p,k) in enumerate(tested):
        adj=min(1.0,max(running,(m-i)*p)); running=adj
        reject=(not stop) and adj<=alpha
        stop=stop or not reject
        out[k].update(p_holm=adj,**{'reject_at_0.05':reject})
    return out

def summarize(comparison_path,ci_path,reference_relations,treatment_relations,protocol_path,output):
    comp=read_json(comparison_path);ci=read_json(ci_path)
    a,b=read_json(reference_relations),read_json(treatment_relations)
    protocol=read_json(protocol_path)
    if protocol.get('status')!='frozen':raise ValueError('Frozen protocol required')
    if a.get('reference_sha256')!=b.get('reference_sha256') or not a.get('reference_sha256'):
        raise ValueError('Relation reports need the identical independent reference inventory')
    aa={str(x['id']):x for x in a['per_image']};bb={str(x['id']):x for x in b['per_image']}
    if set(aa)!=set(bb):raise ValueError('Paired relation evaluation requires all same images')
    official={}
    for key,row in comp['metrics'].items():
        interval=ci.get('metrics',{}).get(key,{}).get('ci95_pp')
        if interval is not None and abs(ci['metrics'][key]['delta_pp']-row['delta_pp'])>1e-8:
            raise ValueError('CI point estimate differs from comparison')
        passed=row['delta_pp']>0 if key=='generative.Cover' else row['observed_non_degradation']
        official[key]=dict(row,ci95_pp=interval,status=('pass' if passed else 'fail') if interval is not None else 'unavailable')
    primary=protocol['registry']['relation_qa_primary_metric']['value']
    if primary!='not_claimed':
        key='relation.'+primary
        if key not in official:raise ValueError('Declare Accuracy/Precision/Recall/F1 or not_claimed')
        official[key]['status']='pass' if official[key]['delta_pp']>0 and official[key]['ci95_pp'] is not None else 'fail' if official[key]['ci95_pp'] is not None else 'unavailable'
    cats=('spatial','relation','action','spatial_and_relation')
    def aggregate(rows,cat,metric):
        numerator='supported' if metric=='precision' else 'matched'
        denominator='predicted' if metric=='precision' else 'reference'
        n=sum(r[cat][numerator] for r in rows);d=sum(r[cat][denominator] for r in rows)
        return n/d if d else None
    rng=random.Random(ci['seed']);names=sorted(aa);reps=ci['repetitions']
    diffs={(c,m):[] for c in cats for m in ('precision','recall')}
    for _ in range(reps):
        ids=rng.choices(names,k=len(names))
        for (cat,m),v in diffs.items():
            x=aggregate([aa[i] for i in ids],cat,m);y=aggregate([bb[i] for i in ids],cat,m)
            if x is not None and y is not None:v.append(y-x)
    relational={}
    for (cat,m),v in diffs.items():
        x=a['metrics'][cat][m];y=b['metrics'][cat][m]
        d=y-x if x is not None and y is not None else None
        v.sort(); interval=[v[int((len(v)-1)*.025)],v[int((len(v)-1)*.975)]] if v else None
        from .amber import bootstrap_p
        relational[cat+'.'+m]={'reference':x,'treatment':y,'delta':d,'ci95':interval,'p_two_sided':bootstrap_p(v),
            'defined_replicates':len(v),'status':'unavailable' if d is None or interval is None else 'pass' if (d>0 if m=='recall' else d>=0) else 'fail'}

    official_p={k:v.get('p_two_sided') for k,v in ci.get('metrics',{}).items()}
    primary={k:(official_p.get(k) if k.startswith('generative.') else relational[k[len('relation.'):]]['p_two_sided'])
             for k in PRIMARY_ENDPOINTS}
    primary_holm=holm(primary)
    gates=[v['status'] for v in official.values()]+[relational['spatial_and_relation.'+m]['status'] for m in ('precision','recall')]
    status='incomplete' if len(official)!=32 or 'unavailable' in gates else 'observed_pass' if all(s=='pass' for s in gates) else 'observed_fail'
    result={'status':status,'success_reference_id':protocol['registry']['success_reference_id']['value'],
        'official':official,'relation_description':relational,
        'relation_co_primary':'spatial_and_relation recall increases AND precision does not decrease',
        'primary_endpoints_holm':primary_holm,
        'note':'Official gate precision is one decimal percentage point. Observed pass is not proof of statistical non-inferiority; confidence intervals and all dimensions remain visible.'}
    atomic_json(output,result)
    return result
