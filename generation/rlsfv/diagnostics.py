"""Independent human relation assessment and transparent retention diagnostics."""
from collections import Counter
from pathlib import Path
import random
from .data import results
from .io import atomic_json, read_json, fingerprint

REL_CATEGORIES=("spatial","relation","action")

def reference_template(manifest_rows, output):
    atomic_json(output,{"instructions":"Annotate visible relations from each image BEFORE viewing model captions. Use stable IDs and entity-aligned atomic statements. Two annotators should adjudicate disagreements. No model verdict is ground truth.",
        "images":[{"id":r["id"],"image":r["image"],"reference_complete":False,"annotators":[],"references":[]} for r in manifest_rows]})

def annotation_template(run, reference_path, method, output):
    source=read_json(reference_path)
    refs={str(x["id"]):x for x in source["images"]}
    images=[]
    for r in results(run):
        image_id=str(r["record"]["id"])
        if image_id not in refs or not refs[image_id]["reference_complete"]:
            raise ValueError("Complete independent references for every image before exposing predictions")
        cid=r["ranking"]["selections"][method]["chosen"]
        c=next(x for x in r["candidates"] if x["id"]==cid)
        predictions=[]
        for key in c["claim_ids"]:
            f=r["facts"][key]
            if f["category"] in REL_CATEGORIES and f["modality"]=="definite":
                predictions.append({"id":key,"category":f["category"],"quote":f["quote"],
                    "statement":{k:f[k] for k in ("subject","predicate","object","scope","polarity","entities")},
                    "verdict":None,"match_reference_ids":[]})
        images.append(dict(refs[image_id],caption=c["text"],caption_id=cid,predictions=predictions,
                           prediction_review_complete=False,extraction_missed=[]))
    immutable=[{"id":x["id"],"caption":x["caption"],"references":x["references"],
                "predictions":[{k:v for k,v in p.items() if k not in ("verdict","match_reference_ids")} for p in x["predictions"]]} for x in images]
    atomic_json(output,{"method":method,"immutable_sha256":fingerprint(immutable),"images":images,
        "instructions":"For every definite prediction assign supported/refuted/unclear by independent image inspection. List all matching judgable reference IDs. Do not edit captions/references/extracted predictions. If extraction missed a relation, mark incomplete and resolve extraction before scoring. Matching is one-to-one. Action is reported separately."})

def _matches(edges):
    assigned={}
    def visit(p, seen):
        for r in edges[p]:
            if r in seen: continue
            seen.add(r)
            if r not in assigned or visit(assigned[r],seen):
                assigned[r]=p
                return True
        return False
    for p in edges: visit(p,set())
    return len(assigned)

def relation_metrics(path,output,repetitions=1000,seed=42):
    data=read_json(path); images=data["images"]
    immutable=[{"id":x["id"],"caption":x["caption"],"references":x["references"],
                "predictions":[{k:v for k,v in p.items() if k not in ("verdict","match_reference_ids")} for p in x["predictions"]]} for x in images]
    if fingerprint(immutable)!=data["immutable_sha256"]:
        raise ValueError("Annotation changed immutable captions/references/prediction inventory")
    counts=[]
    if not images or len({str(x['id']) for x in images})!=len(images): raise ValueError("Empty/duplicate image annotations")
    for im in images:
        if not im["reference_complete"] or not im["prediction_review_complete"] or not im["annotators"]:
            raise ValueError("Independent review incomplete")
        if im.get("extraction_missed"):
            raise ValueError("Resolve missing relation predictions before reporting precision/recall")
        refs=im["references"]
        if len({x["id"] for x in refs})!=len(refs): raise ValueError("Duplicate reference IDs")
        if any(x["category"] not in REL_CATEGORIES or type(x["judgable"]) is not bool or not x["statement"] for x in refs):
            raise ValueError("Invalid reference schema")
        pred=im["predictions"]
        if len({p["id"] for p in pred}) != len(pred) or any(p["category"] not in REL_CATEGORIES for p in pred):
            raise ValueError("Duplicate prediction IDs or invalid relation category")
        if any(x["verdict"] not in ("supported","refuted","unclear") for x in pred): raise ValueError("Unreviewed prediction")
        row={"id":im["id"],"extraction_missed":len(im.get("extraction_missed",[]))}
        for cat in REL_CATEGORIES+("spatial_and_relation",):
            cats={cat} if cat!="spatial_and_relation" else {"spatial","relation"}
            truth={x["id"] for x in refs if x["judgable"] and x["category"] in cats}
            ps=[x for x in pred if x["category"] in cats]
            for p in ps:
                allowed={x["id"] for x in refs if x["judgable"] and x["category"]==p["category"]}
                if not set(p["match_reference_ids"])<=allowed or (p["match_reference_ids"] and p["verdict"]!="supported"):
                    raise ValueError("Invalid relation reference match")
            edges={p["id"]:p["match_reference_ids"] for p in ps if p["verdict"]=="supported"}
            row[cat]={"predicted":len(ps),"supported":sum(p["verdict"]=="supported" for p in ps),
                "unclear":sum(p["verdict"]=="unclear" for p in ps),"reference":len(truth),"matched":_matches(edges)}
        counts.append(row)
    def aggregate(sample):
        out={}
        for cat in REL_CATEGORIES+("spatial_and_relation",):
            c=Counter()
            for r in sample: c.update(r[cat])
            c_missed=sum(r.get("extraction_missed",0) for r in sample)
            p=c["supported"]/c["predicted"] if c["predicted"] else None
            recall=c["matched"]/c["reference"] if c["reference"] else None
            out[cat]=dict(c,precision=p,recall=recall,extraction_missed_total=c_missed,
                f1=2*p*recall/(p+recall) if p is not None and recall is not None and p+recall else 0 if p is not None and recall is not None else None)
        return out
    report={"method":data["method"],"reference_sha256":fingerprint([{"id":x["id"],"references":x["references"]} for x in images]),"metrics":aggregate(counts),"per_image":counts,"truth":"independent_manual_annotation"}
    if repetitions:
        rng=random.Random(seed); samples=[aggregate(rng.choices(counts,k=len(counts))) for _ in range(repetitions)]
        report["ci95"]={}
        for cat in report["metrics"]:
            report["ci95"][cat]={}
            for metric in ("precision","recall","f1"):
                v=sorted(s[cat][metric] for s in samples if s[cat][metric] is not None)
                report["ci95"][cat][metric]={"interval":[v[int((len(v)-1)*.025)],v[int((len(v)-1)*.975)]] if v else None,
                    "defined_replicates":len(v),"total":repetitions}
    atomic_json(output,report)
    return report

def retention(run,output):
    report=[]
    for r in results(run):
        snapshots=r["snapshots"]
        transitions=[]
        for t in (0,1):
            supported={k for k,v in snapshots[t].items() if any(c["status"]=="supported" for c in v["checks"])
                       and not any(c["status"]=="refuted" for c in v["checks"])}
            nxt=[c for c in r["candidates"] if c["stage"]==t+1]
            transitions.append({"from_stage":t,"supported_proxy_pool":len(supported),
                "per_candidate":[{"id":c["id"],"retained":len(supported&set(c["claim_ids"])),
                    "new_facts":len(set(c["claim_ids"])-set(snapshots[t]))} for c in nxt]})
        final={}
        byid={c["id"]:c for c in r["candidates"]}
        for method,sel in r["ranking"]["selections"].items():
            ids=set(byid[sel["chosen"]]["claim_ids"])
            final[method]={"facts":len(ids),"by_first_seen":dict(Counter(r["facts"][k]["first_seen"] for k in ids))}

        persistence={}
        for k,f in r["facts"].items():
            checks=f.get("checks") or []
            group=("unverified" if len(checks)!=2 else "both_supported" if all(c["status"]=="supported" for c in checks)
                   else "both_refuted" if all(c["status"]=="refuted" for c in checks) else "mixed")
            if f["counts"][0]>0:
                p=[k_/n for k_,n in zip(f["counts"],(10,7,5))]
                persistence.setdefault(group,[]).append(p)
        report.append({"id":r["record"]["id"],"transitions":transitions,"selected":final,"persistence_by_support":persistence,
                       "facts":[{"id":k,"counts":f["counts"],"presence":f["presence"],"first_seen":f["first_seen"]} for k,f in r["facts"].items()]})
    summary={}
    for row in report:
        for group,ps in row["persistence_by_support"].items():
            summary.setdefault(group,[]).extend(ps)
    persistence_summary={g:{"facts":len(v),"mean_rate_round0":sum(p[0] for p in v)/len(v),
        "mean_rate_round1":sum(p[1] for p in v)/len(v),"mean_rate_round2":sum(p[2] for p in v)/len(v)}
        for g,v in summary.items() if v}
    atomic_json(output,{"warning":"Self-verification and retention are proxies, not ground truth or official AMBER Cover.",
        "persistence_by_support":persistence_summary,
        "persistence_note":"Facts first seen in round 0. If both_refuted facts persist into rounds 1-2 about as often as both_supported ones, the stability weight mainly reflects prompt compliance and carries little truth information.",
        "images":report})

def annotation_from_captions(cfg,backend,answers_path,reference_path,output,method=None):
    """Same independent relation evaluation for single-pass trained model captions.
    A frozen BASE model extracts text facts only; the human reference stays separate.
    """
    from .io import records
    from .pipeline import Pipeline
    from .prompts import extraction
    from .facts import parse_facts, ensure_existence, make_aligner, union
    refs=read_json(reference_path)
    answers=records(answers_path)
    if len({str(a['id']) for a in answers})!=len(answers):raise ValueError('Duplicate answer IDs')
    lookup={str(a['id']):a['response'] for a in answers}
    output=Path(output)
    if output.exists():raise FileExistsError('Choose new annotation output')
    pipe=Pipeline(cfg,backend,None,output.parent/(output.stem+'.extraction'))
    images=[]
    from .io import lock
    with lock(pipe.output):
        for ref in refs['images']:
            if not ref['reference_complete']:raise ValueError('Independent reference incomplete')
            caption=lookup.get(str(ref['id']))
            if caption is None:raise ValueError('Missing diagnostic caption')
            if pipe.hybrid is not None:

                facts=pipe.hybrid.extract(caption)['facts']
            else:
                facts,_=pipe.generate(None,extraction(caption),'claim_extraction','extract-'+fingerprint(caption),lambda t:parse_facts(t,caption))
            c={'id':str(ref['id']),'stage':0,'facts':ensure_existence(facts)}
            fs=union([c],make_aligner(cfg))
            ps=[{'id':k,'category':f['category'],'quote':f['quote'],
                 'statement':{x:f[x] for x in ('subject','predicate','object','scope','polarity','entities')},
                 'verdict':None,'match_reference_ids':[]} for k,f in fs.items()
                if f['category'] in REL_CATEGORIES and f['modality']=='definite']
            images.append(dict(ref,caption=caption,caption_id=c['id'],predictions=ps,prediction_review_complete=False,
                               extraction_missed=[]))
    immutable=[{'id':x['id'],'caption':x['caption'],'references':x['references'],
        'predictions':[{k:v for k,v in p.items() if k not in ('verdict','match_reference_ids')} for p in x['predictions']]} for x in images]
    atomic_json(output,{'method':method or str(answers_path),'immutable_sha256':fingerprint(immutable),'images':images,
        'instructions':'Independent human image review required. Complete verdict/matches without editing the reference or prediction inventory. List relations that the caption states but the prediction inventory lacks in extraction_missed (quote + category); they are reported as extraction recall, not silently added.'})
