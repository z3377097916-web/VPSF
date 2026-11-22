import argparse
import json
from pathlib import Path
from . import config
from .io import read_json, atomic_json

def main():
    p=argparse.ArgumentParser(description="Coverage and relation experiment, revision protocol v1")
    sub=p.add_subparsers(dest="command",required=True)
    def add(name,fields):
        c=sub.add_parser(name)
        for field in fields: c.add_argument("--"+field,required=True)
        return c
    c=add("run",["config","manifest","output"]); c.add_argument("--protocol")
    add("original-baseline",["config","manifest","output"])
    c=add("export",["run","output"]); c.add_argument("--scoring-config")
    add("freeze",["registry","config","training","output"])
    add("preflight",["config"])
    add("prepare-amber",["config","images","output"])
    c=add("infer-images",["config","manifest","output","decoding"]); c.add_argument("--variant",choices=["base","rlsf","trained"],default="base"); c.add_argument("--adapter"); c.add_argument("--protocol")
    c=add("infer",["config","images","output"]); c.add_argument("--variant",choices=["base","rlsf","trained"],default="base"); c.add_argument("--adapter"); c.add_argument("--task",choices=["a","g","d"],default="a"); c.add_argument("--decoding",required=True,help="Explicit JSON decoding settings"); c.add_argument("--protocol")
    c=add("evaluate",["config","answers","output"]); c.add_argument("--task",choices=["a","g"],default="a"); c.add_argument("--protocol")
    add("compare",["reference","treatment","output"])
    add("summarize",["comparison","ci","reference-relations","treatment-relations","protocol","output"])
    c=add("bootstrap",["config","reference","treatment","output"]); c.add_argument("--task",choices=["a","g"],default="a"); c.add_argument("--repetitions",type=int,default=1000); c.add_argument("--seed",type=int,default=42)
    c=add("train",["config","training","pairs","output"]); c.add_argument("--execute",action="store_true"); c.add_argument("--resume",action="store_true"); c.add_argument("--protocol"); c.add_argument("--group")
    add("retention",["run","output"])
    c=add("leakage-check",["output"]); c.add_argument("--manifests",nargs="+",required=True)
    c=add("reference-template",["manifest","output"])
    c.add_argument("--config")
    c.add_argument("--official-vendor")
    add("annotation-template",["run","references","method","output"])
    c=add("annotation-from-captions",["config","answers","references","output"]); c.add_argument("--method")
    c=add("relation-metrics",["annotations","output"]); c.add_argument("--repetitions",type=int,default=1000); c.add_argument("--seed",type=int,default=42)
    for command, fields in (("evaluate-selection", ["config", "selection", "output"]),
                             ("relation-groups", ["selection", "output"])):
        c=add(command, fields)
        c.add_argument("--greedy"); c.add_argument("--annotation-index")
        c.add_argument("--repetitions",type=int,default=1000); c.add_argument("--seed",type=int,default=42)
        if command=="evaluate-selection": c.add_argument("--discriminative")
    add("smoke",["output"])
    c=add("make-dev",["coco-images","instances","amber-images","output"]); c.add_argument("--n",type=int,default=200); c.add_argument("--seed",type=int,default=42); c.add_argument("--download",action="store_true",help="fetch only the sampled images via coco_url")
    add("chair",["answers","instances","captions","output"])
    a=p.parse_args()
    cfg=config.load(a.config) if getattr(a,"config",None) else None
    from . import data, amber, diagnostics
    result=None
    if getattr(a,"protocol",None) and cfg is not None:
        from .governance import verify
        verify(a.protocol,cfg)
    if a.command=="make-dev":
        from .devset import make_manifest
        result=make_manifest(a.coco_images,a.instances,a.amber_images,a.n,a.seed,a.output,a.download)
    elif a.command=="chair":
        from .devset import evaluate as chair_eval
        result=chair_eval(a.answers,a.instances,a.captions,a.output)
    elif a.command=="freeze":
        from .governance import freeze
        result=freeze(a.registry,a.config,a.training,a.output)
    elif a.command=="run":
        from .pipeline_factory import build
        rows=data.manifest(a.manifest, official_amber_vendor=cfg["vendor"])
        result=build(cfg,a.output).run(rows)
    elif a.command=="original-baseline":
        from .backend import PosParser
        from .pipeline_factory import build_backend
        from .baseline import run
        backend=build_backend(cfg,a.output,variant="base")
        result=run(cfg,backend,PosParser(cfg["pos_model"]),data.manifest(a.manifest, official_amber_vendor=cfg["vendor"]),a.output)
    elif a.command=="export":
        s=config.load(a.scoring_config) if a.scoring_config else None
        if s:
            prior=read_json(Path(a.run)/"run.json")["config"]
            if {k:v for k,v in s.items() if k!="scoring"}!={k:v for k,v in prior.items() if k!="scoring"}:
                raise ValueError("Rescoring may change only scoring fields; prompt/model/sampling changes require a new generation run")
        result=data.export(a.run,a.output,s["scoring"] if s else None,s["seed"] if s else None)
    elif a.command=="prepare-amber":
        rows=[{"id":q["id"],"image":str((Path(a.images)/q["image"]).resolve()),"split":"test","source":"AMBER"} for q in amber.queries(cfg["vendor"],"g")]
        atomic_json(a.output,rows); result={"images":len(rows)}
    elif a.command=="infer":
        from .pipeline_factory import build_backend
        backend=build_backend(cfg,a.output,variant=a.variant,trained_adapter=a.adapter)
        amber.infer(cfg,backend,a.images,a.output,a.task,read_json(a.decoding))
    elif a.command=="infer-images":
        from .pipeline_factory import build_backend
        backend=build_backend(cfg,a.output,variant=a.variant,trained_adapter=a.adapter)
        amber.infer_images(cfg,backend,data.manifest(a.manifest, official_amber_vendor=cfg["vendor"]),a.output,read_json(a.decoding))
    elif a.command=="evaluate":
        result=amber.evaluate(a.answers,a.output,cfg["vendor"],a.task)
        result["experiment_status"]="frozen_protocol" if a.protocol else "development_only"
        result["protocol"] = a.protocol
        atomic_json(Path(a.output)/"report.json",result)
    elif a.command=="compare": result=amber.compare(a.reference,a.treatment,a.output)
    elif a.command=="summarize":
        from .reporting import summarize
        result=summarize(a.comparison,a.ci,a.reference_relations,a.treatment_relations,a.protocol,a.output)
    elif a.command=="bootstrap": result=amber.bootstrap(a.reference,a.treatment,cfg["vendor"],a.output,a.repetitions,a.seed,a.task)
    elif a.command=="train":
        if cfg.get('backend', 'llava') != 'llava':
            raise ValueError('10.1 Qwen is generation/selection/evaluation only; Qwen LoRA-DPO is not implemented')
        from . import train
        tc=train.config(a.training)
        if a.execute:
            if not a.protocol or not a.group:
                p.error("Training execution requires --protocol and --group; freeze required decisions first")
            from .governance import verify
            verify(a.protocol,cfg,tc,data.validate_pairs(a.pairs),a.group)
        result=train.run(cfg,tc,a.pairs,a.output,a.resume) if a.execute else train.plan(cfg,tc,a.pairs,a.output)
    elif a.command=="retention": diagnostics.retention(a.run,a.output)
    elif a.command=="leakage-check": result=data.leakage_check(a.manifests,a.output)
    elif a.command=="reference-template":
        vendor = a.official_vendor or (cfg["vendor"] if cfg else None)
        diagnostics.reference_template(data.manifest(a.manifest, official_amber_vendor=vendor),a.output)
    elif a.command=="annotation-template": diagnostics.annotation_template(a.run,a.references,a.method,a.output)
    elif a.command=="annotation-from-captions":
        from .pipeline_factory import build_backend
        backend=build_backend(cfg,str(Path(a.output).parent/"annotation-model"),variant="base")
        diagnostics.annotation_from_captions(cfg,backend,a.answers,a.references,a.output,a.method)
    elif a.command=="relation-metrics": result=diagnostics.relation_metrics(a.annotations,a.output,a.repetitions,a.seed)
    elif a.command=="relation-groups":
        from .selection_evaluation import relation_groups
        result=relation_groups(a.selection,a.output,a.annotation_index,a.greedy,a.repetitions,a.seed)
    elif a.command=="evaluate-selection":
        from .selection_evaluation import evaluate_groups
        result=evaluate_groups(cfg,a.selection,a.output,a.discriminative,a.greedy,a.annotation_index,a.repetitions,a.seed)
    elif a.command=="smoke":
        from .smoke import run
        result=run(a.output)
    elif a.command=="preflight":
        from .preflight import inspect_environment
        result=inspect_environment(cfg)
        if not result['passed']:
            print(json.dumps(result,indent=2)); raise SystemExit(2)
    if result is not None: print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
