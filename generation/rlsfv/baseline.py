"""Controlled original-style baseline: ten x0 captions, author native-logit U.
Uses the vendored author's fuzzy and POS functions, including their degenerate
row convention. It is separate from revised common-context scoring.
"""
import importlib.util
from pathlib import Path
import random
from .config import X0
from .facts import caption_valid, norm
from .io import Store, atomic_json, fingerprint, sha256, read_json, code_hash, lock

def run(cfg,backend,parser,rows,output):
    qwen = cfg.get('backend', 'llava') in {'qwen', 'qwen2.5', 'qwen25', 'qwen3', 'qwen3-vl'}
    relative = ('RLSF-V/train/hf_series/generate/convert_qwen.py' if qwen
                else 'RLSF-V/train/llava_series/generate/convert_llava.py')
    source=Path(cfg['vendor'])/relative
    spec=importlib.util.spec_from_file_location('rlsf_author_convert',source)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    author.worker_tokenizer=backend.tokenizer;author.worker_nlp=parser.nlp
    torch=backend.torch
    output=Path(output)
    identity={'cfg':cfg,'model':backend.meta,'rows':rows,'code':code_hash(),
              'author_source_sha256':sha256(source),'n_candidates':10,'native_scores':True}
    with lock(output):
        if (output/'run.json').exists() and read_json(output/'run.json')!=identity:
            raise ValueError('Baseline run identity changed')
        atomic_json(output/'run.json',identity)
        store=Store(output,identity);pairs=[];answers=[];audit=[];failures=[]
        for record in rows:
            try:
                candidates=[]; image=record['image']; ih=sha256(image)
                for i in range(10):
                    seed=int(fingerprint([cfg['seed'],record['id'],i])[:8],16)
                    req={'kind':'original_caption','image_sha256':ih,'prompt':X0,'settings':cfg['generation'],
                         'seed':seed,'slot':i,'history':[]}
                    r=store.cached(req,lambda:backend.generate(image,X0,cfg['generation'],seed,capture_scores=True))
                    if r['hit_limit']:raise ValueError('Truncated original candidate; retained for review')
                    caption_valid(r['text'])
                    vals=torch.tensor([[float('-inf') if v is None else v for v in row] for row in r['native_topk_values']])
                    if len(vals)!=len(r['token_ids']):raise ValueError('Native score/token length mismatch')
                    entropies=author.fuzzy_minmax_entropy_from_logits(vals).tolist()
                    if qwen:
                        mask,_,_=author.get_key_token_mask(r['token_ids'])
                    else:
                        mask,_=author.get_key_token_mask(backend.tokenizer.convert_ids_to_tokens(r['token_ids']))
                    chosen_h=sorted([v for v,m in zip(entropies,mask) if m],reverse=True)[:cfg['fuzzy']['top_r']]
                    u=sum(chosen_h)/len(chosen_h) if chosen_h else 0.0
                    candidates.append({'id':i,'text':r['text'],'u':u,'raw':r,'entropies':entropies,'mask':mask})
                rng=random.Random(int(fingerprint([cfg['seed'],record['id'],'original_ties'])[:16],16))
                order=candidates[:];rng.shuffle(order)
                best=min(order,key=lambda c:c['u']);different=[c for c in order if norm(c['text'])!=norm(best['text'])]
                worst=max(different,key=lambda c:c['u']) if different else None
                valid=worst is not None and worst['u']>best['u']
                answers.append({'id':record['id'],'response':best['text']})
                if valid and record['split']=='train':
                    pairs.append({'image_id':record['id'],'img_path':image,'image_sha256':ih,'split':'train','prompt':X0,
                                  'chosen':best['text'],'rejected':worst['text']})
                audit.append({'id':record['id'],'pair_valid':valid,'chosen':best['id'],'rejected':worst['id'] if worst else None})
                atomic_json(output/'images'/(fingerprint(record['id'])+'.json'),{'record':record,'candidates':candidates,'selection':audit[-1]})
            except Exception as exc:
                failures.append({'id':record['id'],'error':repr(exc)})
            print(f"Original baseline {len(audit)+len(failures)}/{len(rows)}; failed {len(failures)}",flush=True)
        atomic_json(output/'status.json',{'complete':not failures,'failures':failures,'images':len(rows)})
        atomic_json(output/'audit.json',audit)
        if failures:raise ValueError('Original baseline incomplete. Failed images retained; no complete export.')
        atomic_json(output/'pairs.json',pairs);atomic_json(output/'captions.json',answers)
        return {'images':len(rows),'pairs':len(pairs),'native_author_score':True}
