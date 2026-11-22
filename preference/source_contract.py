"""Bind exported features and training pairs to their recorded prompt and model."""
import hashlib
import json
from pathlib import Path
from .protocol import PROMPT

VERSION = 'e3_original_llava_source_v1'
ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    'base': ('config.json', 'generation_config.json', 'pytorch_model-00001-of-00002.bin',
             'pytorch_model-00002-of-00002.bin', 'pytorch_model.bin.index.json',
             'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json'),
    'vision': ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def expected_model():
    manifest=json.loads((ROOT/'runtime/model_manifest.json').read_text())
    files=[]
    for group,names in MODEL_FILES.items():
        spec=manifest[group];entries={e['name']:e for e in spec['files']}
        for name in names:
            e=entries[name];sha=(e.get('lfs') or {}).get('sha256') or e.get('official_sha256')
            if not sha:raise ValueError('Missing authoritative model hash: '+name)
            files.append(dict(group=group,name=name,revision=spec['sha'],sha256=sha))
    return dict(backend='llava',variant='base',files=files)


def validate_source(source):
    if not isinstance(source,dict) or source.get('version')!=VERSION:
        raise ValueError('Missing verified source contract; export complete raw records again')
    if source.get('prompt')!=PROMPT:raise ValueError('Generation/training prompt mismatch')
    if source.get('model')!=expected_model():raise ValueError('Require original LLaVA-v1.5-7B and original vision weights')
    if source.get('feedback_backend')!='same_original_model':raise ValueError('Require same-model self-feedback')
    return source


def raw_source(record,run_identity=None):
    identity=record.get('generation_identity') or run_identity
    if not isinstance(identity,dict):raise ValueError('Raw record lacks generation model identity; use --diagnostic for historical data')
    if run_identity is not None and record.get('generation_identity') is not None and identity!=run_identity:
        raise ValueError('Raw generation identity differs from run.json')
    cfg=identity.get('config',{});meta=identity.get('model',{})
    if cfg.get('backend','llava')!='llava' or cfg.get('extraction',{}).get('mode') not in {'original','structured_sentences','hybrid'}:
        raise ValueError('Require original LLaVA for generation and text adjudication')
    if meta.get('variant')!='base' or meta.get('adapter_sha256')!={}:
        raise ValueError('Require base model without an adapter for self-feedback')
    approved=expected_model();inventory=meta.get('model_inventory',[])
    lookup={}
    for r in inventory:
        key=(r.get('group'),r.get('name'))
        if key in lookup:raise ValueError('Duplicate model inventory entry')
        lookup[key]=r
    for spec in approved['files']:
        found=lookup.get((spec['group'],spec['name']),{})
        if any(found.get(k)!=v for k,v in spec.items()):raise ValueError('Generation model inventory is incomplete or differs from approved weights')
    candidates=record.get('candidates',[])
    fresh=[c for c in candidates if c.get('branch')=='fresh']
    if len(fresh)!=12 or any(c.get('branch')!='fresh' for c in candidates if c.get('stage')==0):
        raise ValueError('Expected 10+1+1 fresh candidates')
    if any(c.get('prompt')!=PROMPT for c in fresh):raise ValueError('Actual fresh/stage-0 prompt differs from training prompt')
    if any(c.get('fuzzy',{}).get('common_prompt')!=PROMPT for c in candidates):
        raise ValueError('Missing or inconsistent common scoring prompt; historical records are diagnostic only')
    if any(c.get('fuzzy',{}).get('history_messages',0) for c in candidates):raise ValueError('History-contaminated scoring')
    source=dict(version=VERSION,prompt=PROMPT,model=approved,feedback_backend='same_original_model')
    return source,fingerprint(identity)


def bundle_source(directory,feats,paths):
    path=Path(directory)/'feature_contract.json'
    if not path.is_file():raise ValueError('Missing feature_contract.json; historical compact features are diagnostic only')
    contract=json.loads(path.read_text())
    if contract.get('usage')!='training' or contract.get('schema')!='legacy_coco_v1':
        raise ValueError('Diagnostic/unknown feature bundle cannot enter training')
    source=validate_source(contract.get('source'))
    if contract.get('images')!=len(feats):raise ValueError('Feature image count differs from contract')
    hashes=contract.get('files',{})
    if set(hashes)!={p.name for p in paths}:raise ValueError('Feature contract file list differs')
    for p in paths:
        if file_hash(p)!=hashes[p.name]:raise ValueError('Feature file changed after export: '+p.name)
    runs=set(contract.get('source_runs',[]))
    for im in feats:
        if im.get('source_contract')!=source or im.get('source_run_hash') not in runs:
            raise ValueError('Feature source differs from its bundle contract')
    return source,path
