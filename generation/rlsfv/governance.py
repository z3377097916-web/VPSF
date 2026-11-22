"""Explicit protocol freezing. Development runs are not formal experiment results."""
from datetime import datetime, timezone
from pathlib import Path
from .io import read_json, atomic_json, sha256, fingerprint
from .config import load
from .data import leakage_check
from .train import config as training_config

REQUIRED={"training_source","train_manifest","dev_manifest","test_manifest","diagnostic_manifest",
          "near_duplicate_audit","groups","diagnostic_plan","budget","success_reference_id","relation_qa_primary_metric"}

def freeze(registry_path,config_path,training_path,output):
    registry=read_json(registry_path)
    if set(registry)!=REQUIRED: raise ValueError(f"Registry requires exactly {sorted(REQUIRED)}")
    for key,entry in registry.items():
        if set(entry)!={"status","value"} or entry["status"] not in {"confirmed","proposed","required"}:
            raise ValueError(f"Invalid configuration state: {key}")
        if entry["status"]!="confirmed" or entry["value"] is None:
            raise ValueError(f"CONFIG_REQUIRED: resolve and confirm {key} before formal freezing")
    base=Path(registry_path).resolve().parent
    paths=[str((base/registry[k]["value"]).resolve()) for k in ("train_manifest","dev_manifest","test_manifest","diagnostic_manifest")]

    output=Path(output)
    if output.exists():raise FileExistsError("Frozen protocol cannot be overwritten")
    audit_path=base/registry["near_duplicate_audit"]["value"]
    audit=read_json(audit_path)
    if audit.get("status")!="reviewed" or audit.get("unresolved_pairs")!=0 or not audit.get("method") or not audit.get("reviewer"):
        raise ValueError("Near-duplicate audit must be reviewed, with method/reviewer and zero unresolved pairs")
    expected=sorted(sha256(p) for p in paths)
    if sorted(audit.get("manifest_sha256",[]))!=expected:
        raise ValueError("Near-duplicate audit does not refer to these four manifests")
    groups=registry["groups"]["value"]
    allowed={"original","B0","Bf","Bs","Bq","Ball","BqF","G-B0","G-Bf","G-Bs","G-Bq","G-Ball","G-BqF"}
    if not isinstance(groups,list) or not groups or len(set(groups))!=len(groups) or not set(groups)<=allowed:
        raise ValueError("Explicit unique training groups required")

    from .data import manifest
    for path,split in zip(paths,("train","dev","test","diagnostic")):
        if any(r["split"]!=split for r in manifest(path)):raise ValueError(f"Manifest role mismatch: expected {split}")
    report=leakage_check(paths,output.parent/(output.stem+'.exact-leakage.json'))
    payload={"status":"frozen","frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "registry":registry,"config":load(config_path),"training":training_config(training_path),
        "manifests":[{"path":p,"sha256":sha256(p)} for p in paths],
        "near_duplicate_audit_sha256":sha256(audit_path),"images":report["images"]}
    atomic_json(output,payload)
    return {"status":"frozen","file":str(output),"images":len(report["images"]),"groups":groups}

def verify(path,cfg,training=None,pairs=None,group=None,stage=None):
    p=read_json(path)
    if p.get("status")!="frozen" or p["config"]!=cfg: raise ValueError("Config differs from frozen protocol")
    if training is not None and p["training"]!=training: raise ValueError("Training config differs from frozen protocol")
    if group is not None and group not in p["registry"]["groups"]["value"]: raise ValueError("Group not in frozen training budget")
    if pairs is not None:
        inventory={x["sha256"] for x in p["images"] if x["split"]=="train"}
        if any(x["image_sha256"] not in inventory for x in pairs):raise ValueError("Training pairs outside frozen train-image inventory")
    return p
