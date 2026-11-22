"""Run the vendored official AMBER implementation without changing scoring rules."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from .io import atomic_json, read_json, records, sha256, Store, lock, fingerprint, code_hash

SECTIONS = {"Generative Task": "generative", "Descriminative Task": "discriminative", "Exsitence": "existence",
            "Attribute": "attribute", "State": "state", "Number": "number", "Action": "action", "Relation": "relation"}

def queries(vendor, task="a"):
    if task not in {'a','g','d'}:
        raise ValueError('Unknown AMBER task')
    filename = "query_all.json" if task in {"a","d"} else "query_generative.json"
    data = read_json(Path(vendor)/"AMBER/data/query"/filename)
    n = 15220 if task in {"a","d"} else 1004
    if len(data) != n or {x["id"] for x in data} != set(range(1,n+1)):
        raise ValueError("Unexpected official query inventory")
    return [q for q in data if q['id']>1004] if task=='d' else data

def combine_frozen_answers(captions, discriminative, vendor):
    """Combine disjoint official task IDs; Yes/No answers are a shared baseline.

    This never fabricates per-selection-method relation improvements.
    """
    g,_=normalize(captions,vendor,'g')
    d,_=normalize(discriminative,vendor,'d')
    combined=sorted(g+d,key=lambda row:row['id'])
    if len(combined)!=15220 or {x['id'] for x in combined}!={q['id'] for q in queries(vendor,'a')}:
        raise ValueError('Incomplete combined official answer inventory')
    return combined

def yes_no(text):
    m = re.fullmatch(r"\s*(yes|no)[.!]?\s*", text, re.I)
    return m[1].title() if m else None

def normalize(path, vendor, task):
    data = records(path)
    expected = {q["id"] for q in queries(vendor,task)}
    if len(data) != len(expected) or {q["id"] for q in data} != expected:
        raise ValueError("Official answers must contain all expected IDs exactly once")
    result, invalid = [], 0
    for r in sorted(data, key=lambda r:r["id"]):
        if type(r["id"]) is not int or not isinstance(r["response"],str):
            raise ValueError("Expected {id: integer, response: string}")
        response = r["response"]
        if r["id"] > 1004:
            val = yes_no(response)
            invalid += val is None
            response = val if val else response
        result.append({"id": r["id"], "response": response})
    return result, invalid

def parse_metrics(text, task="a"):
    result, section = {}, None
    for line in text.splitlines():
        clean = line.strip()
        if clean.rstrip(":") in SECTIONS:
            section = SECTIONS[clean.rstrip(":")]
            continue
        m = re.fullmatch(r"(CHAIR|Cover|Hal|Cog|Accuracy|Precision|Recall|F1):\s*([0-9]+(?:\.[0-9]+)?)",clean)
        if m and section:
            key = section+"."+m[1]
            if key in result:
                raise ValueError("Duplicate official metric")
            result[key] = float(m[2])
    expected = {"generative."+x for x in ("CHAIR","Cover","Hal","Cog")}
    if task == "a":
        expected |= {s+"."+m for s in SECTIONS.values() if s != "generative" for m in ("Accuracy","Precision","Recall","F1")}
    if set(result) != expected:
        raise ValueError(f"Official metrics incomplete: expected {len(expected)}, got {len(result)}")
    return result

def args_for(vendor, answer_path, task):
    root = Path(vendor).resolve()/"AMBER"
    return SimpleNamespace(inference_data=str(answer_path), evaluation_type=task,
        annotation=str(root/"data/annotations.json"), word_association=str(root/"data/relation.json"),
        safe_words=str(root/"data/safe_words.txt"), metrics=str(root/"data/metrics.txt"), similarity_score=0.8)

def evaluate(answers, output, vendor, task="a"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose new evaluation directory")
    data, invalid = normalize(answers,vendor,task)
    atomic_json(output/"answers.json", data)
    args = args_for(vendor,output.resolve()/"answers.json",task)
    source = Path(vendor).resolve()/"AMBER/inference.py"
    cmd = [sys.executable,str(source)]
    for k,v in vars(args).items():
        cmd += ["--"+k,str(v)]
    report = {"status":"running","task":task,"n":len(data),"invalid_yes_no":invalid,
        "invalid_rate":invalid/14216 if task=="a" else None,"source_sha256":sha256(source),
        "input_sha256":sha256(answers),"official_assets":{k:sha256(getattr(args,k)) for k in ("annotation","word_association","safe_words","metrics")},
        "command":cmd, "positive_class":"No"}
    atomic_json(output/"report.json",report)
    with (output/"official.log").open("w") as log:
        p = subprocess.run(cmd,cwd=source.parent,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:
        report.update(status="failed",exit_code=p.returncode)
        atomic_json(output/"report.json",report)
        raise RuntimeError(f"Official AMBER failed: {output/'official.log'}")
    report.update(status="complete", metrics_percent=parse_metrics((output/"official.log").read_text(),task))
    atomic_json(output/"report.json",report)
    return report

def infer(cfg, backend, image_root, output, task="a", settings=None):
    output=Path(output)
    q=queries(cfg["vendor"],task)
    settings=settings or {"temperature":0,"top_p":1,"top_k":50,"max_new_tokens":1024}
    identity={"model":backend.meta,"settings":settings,"code":code_hash(),"queries":fingerprint(q),"image_root":str(Path(image_root).resolve())}
    with lock(output):
        if (output/"run.json").exists() and read_json(output/"run.json")!=identity:
            raise ValueError("Inference identity changed")
        atomic_json(output/"run.json",identity)
        store=Store(output,identity)
        raw, answers, errors=[],[],[]
        for i,item in enumerate(q):
            image=Path(image_root)/item["image"]
            try:
                req={"kind":"standard_inference","image_sha256":sha256(image),"prompt":item["query"],"id":item["id"],"seed":cfg["seed"]}
                r=store.cached(req,lambda:backend.generate(str(image),item["query"],settings,cfg["seed"]))
                if r["hit_limit"]:
                    raise ValueError("Truncated evaluation answer")
                raw.append(dict(item,result=r))
                response=r["text"]
                answers.append({"id":item["id"],"response":yes_no(response) or response if item["id"]>1004 else response})
            except Exception as exc:
                errors.append({"id":item["id"],"error":repr(exc)})
            if i%50==0:
                print(f"AMBER {i+1}/{len(q)}; failed {len(errors)}",flush=True)
        atomic_json(output/"raw.json",raw)
        atomic_json(output/"errors.json",errors)
        atomic_json(output/"answers.json",answers)
        atomic_json(output/"status.json",{"complete":not errors,"expected":len(q),"answers":len(answers)})
        if errors:
            raise ValueError("Incomplete AMBER inference retained; resume before evaluation")

def compare(reference, treatment, output):
    a,b=read_json(reference),read_json(treatment)
    if a["status"]!="complete" or b["status"]!="complete" or a["task"]!=b["task"]:
        raise ValueError("Comparable complete official reports required")
    if a["source_sha256"]!=b["source_sha256"] or a["official_assets"]!=b["official_assets"]:
        raise ValueError("Different official evaluator versions/assets")
    rows={}
    for k,x in a["metrics_percent"].items():
        delta=b["metrics_percent"][k]-x
        favorable=-delta if k in {"generative.CHAIR","generative.Hal","generative.Cog"} else delta
        rows[k]={"reference":x,"treatment":b["metrics_percent"][k],"delta_pp":delta,"observed_non_degradation":favorable>=0}
    result={"reference":str(reference),"treatment":str(treatment),"metrics":rows,
        "all_observed_non_degradation":all(x["observed_non_degradation"] for x in rows.values()),
        "cover_improved":rows["generative.Cover"]["delta_pp"]>0,
        "relation_caption_status":"REQUIRES_INDEPENDENT_ANNOTATION",
        "note":"Observed values are rounded official metrics; statistical non-inferiority is not established."}
    atomic_json(output,result)
    return result

def bootstrap_p(deltas):
    """Two-sided paired-bootstrap p-value: 2 * min(P(delta <= 0), P(delta >= 0)), capped at 1."""
    if not deltas:
        return None
    n = len(deltas)
    return min(1.0, 2 * min(sum(d <= 0 for d in deltas) / n, sum(d >= 0 for d in deltas) / n))

def bootstrap(reference, treatment, vendor, output, repetitions, seed, task="a"):
    """Paired image-cluster resampling through ORIGINAL main(), not a rewritten scorer.
    Slow but exact, including epsilon initialization, rounding and No-positive logic.
    """
    if repetitions<2:
        raise ValueError("At least two bootstrap samples required")
    a,_=normalize(reference,vendor,task); b,_=normalize(treatment,vendor,task)
    q=queries(vendor,task)
    clusters={}
    for item in q:
        clusters.setdefault(item["image"],[]).append(item["id"])
    amap,bmap={x["id"]:x for x in a},{x["id"]:x for x in b}
    source=Path(vendor).resolve()/"AMBER/inference.py"
    spec=importlib.util.spec_from_file_location("official_amber_bootstrap",source)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    from functools import lru_cache
    mod.extract_nouns=lru_cache(maxsize=None)(mod.extract_nouns)
    mod.check_synonyms_word=lru_cache(maxsize=None)(mod.check_synonyms_word)
    mod.tqdm=lambda x:x
    rng=random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="amber-bootstrap-") as tmp:
        path=Path(tmp)/"sample.json"
        mod.args=args_for(vendor,path,task)
        def run(data):
            atomic_json(path,data)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                mod.main(mod.args)
            return parse_metrics(stream.getvalue(),task)
        point_a,point_b=run(a),run(b)
        deltas={k:[] for k in point_a}
        names=sorted(clusters)
        for it in range(repetitions):
            ids=[idx for name in rng.choices(names,k=len(names)) for idx in clusters[name]]
            ma,mb=run([amap[x] for x in ids]),run([bmap[x] for x in ids])
            for k in deltas:
                deltas[k].append(mb[k]-ma[k])
            if it%10==0: print(f"bootstrap {it+1}/{repetitions}",flush=True)
    def quantile(values,p):
        v=sorted(values); x=(len(v)-1)*p; lo=int(x); hi=min(lo+1,len(v)-1)
        return v[lo]+(v[hi]-v[lo])*(x-lo)
    result={"paired_unit":"image","repetitions":repetitions,"seed":seed,"task":task,"source_sha256":sha256(source),
        "metrics":{k:{"delta_pp":point_b[k]-point_a[k],"ci95_pp":[quantile(v,.025),quantile(v,.975)],
                      "p_two_sided":bootstrap_p(v)} for k,v in deltas.items()},
        "note":"Percentile intervals over official rounded statistics. p_two_sided is the paired-bootstrap sign proportion; multiplicity correction for the pre-registered primary endpoints is applied in reporting.summarize (Holm)."}
    atomic_json(output,result)
    return result

def infer_images(cfg,backend,rows,output,settings):
    """Standard single-pass x0 captions on the separate diagnostic image set."""
    from .config import X0
    from .facts import caption_valid
    output=Path(output)
    identity={'model':backend.meta,'rows':rows,'settings':settings,'code':code_hash(),'prompt':X0}
    with lock(output):
        if (output/'run.json').exists() and read_json(output/'run.json')!=identity:
            raise ValueError('Diagnostic inference identity changed')
        atomic_json(output/'run.json',identity)
        store=Store(output,identity);answers=[];errors=[]
        for row in rows:
            try:
                req={'kind':'diagnostic_single_pass','image_sha256':sha256(row['image']),'prompt':X0,'seed':cfg['seed'],'history':[]}
                result=store.cached(req,lambda:backend.generate(row['image'],X0,settings,cfg['seed']))
                if result['hit_limit']:raise ValueError('Truncated diagnostic caption')
                answers.append({'id':row['id'],'response':caption_valid(result['text'])})
            except Exception as exc:errors.append({'id':row['id'],'error':repr(exc)})
        atomic_json(output/'answers.json',answers)
        atomic_json(output/'status.json',{'complete':not errors,'expected':len(rows),'errors':errors})
        if errors:raise ValueError('Diagnostic inference incomplete; resume, do not drop images')
