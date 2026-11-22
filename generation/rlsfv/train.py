"""LoRA DPO on the verified original LLaVA; fixed reference = disabled adapters.
No model-weight updates are performed by any candidate-generation command.
"""
import math
import os
from pathlib import Path
from .io import atomic_json, read_json, sha256, fingerprint, code_hash, lock
from .data import validate_pairs

REQUIRED = {"seed", "epochs", "learning_rate", "beta", "lora_r", "lora_alpha", "lora_dropout",
            "batch_size", "gradient_accumulation_steps", "weight_decay", "warmup_ratio",
            "max_grad_norm", "precision", "max_length", "gradient_checkpointing", "save_every_steps"}

def config(path):
    c=read_json(path)
    if set(c)!=REQUIRED or any(v is None for v in c.values()):
        raise ValueError(f"Explicit training values required, exactly: {sorted(REQUIRED)}")
    for k in ("seed","epochs","lora_r","lora_alpha","batch_size","gradient_accumulation_steps","max_length","save_every_steps"):
        if type(c[k]) is not int or c[k]< (0 if k=="seed" else 1):
            raise ValueError(f"Invalid training {k}")
    if c["precision"] not in ("bf16","fp16") or not 0<=c["lora_dropout"]<1 or not 0<=c["warmup_ratio"]<1:
        raise ValueError("Invalid precision/dropout/warmup")
    if any(c[k]<=0 for k in ("learning_rate","beta","max_grad_norm")) or c["weight_decay"]<0:
        raise ValueError("Invalid optimization values")
    return c

def plan(cfg, training, pairs, output):
    if cfg.get('backend', 'llava') != 'llava':
        raise ValueError('10.1 Qwen training is not implemented; refusing to train LLaVA for a Qwen config')
    rows=validate_pairs(pairs)
    value={"status":"planned_not_trained","n_pairs":len(rows),"pairs_sha256":sha256(pairs),
           "start":"original_LLaVA","reference":"same_frozen_base_with_adapter_disabled",
           "trainable":"language_model_LoRA_only", "training":training,
           "effective_batch_per_device":training["batch_size"]*training["gradient_accumulation_steps"],
           "models":cfg["models"],"vendor":cfg["vendor"],"code":code_hash()}
    atomic_json(output,value)
    return value

def run(cfg, tc, pairs_path, output, resume=False):
    if cfg.get('backend', 'llava') != 'llava':
        raise ValueError('10.1 Qwen training is not implemented; refusing to train LLaVA for a Qwen config')


    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0"))
    local=int(os.environ.get("LOCAL_RANK","0"))
    if world>1:
        visible=os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"]=visible.split(",")[local] if visible else str(local)
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from .backend import LlavaBackend, ContextOverflow
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(0)
    if world>1: dist.init_process_group("nccl")
    out=Path(output).resolve()
    rows=validate_pairs(pairs_path)
    identity={"training":tc,"pairs_sha256":sha256(pairs_path),"code":code_hash(),"world_size":world,"config":cfg}
    if out.exists() and (out/"identity.json").exists():
        if not resume or read_json(out/"identity.json")!=identity:
            raise ValueError("Existing training run requires --resume with identical inputs/config/code/world size")
        if (out/"complete.json").exists(): raise ValueError("Training already completed")
    elif resume: raise ValueError("No run to resume")

    with lock(out/f"rank-{rank}"):
        if rank==0: atomic_json(out/"identity.json",identity)
        random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"]); torch.cuda.manual_seed_all(tc["seed"])
        backend=LlavaBackend(cfg["models"],out/f"rank-{rank}",cfg["vendor"],variant="base")
        if tc["max_length"]>backend.context_len:
            raise ValueError("Training max_length exceeds verified model context")
        dtype=torch.bfloat16 if tc["precision"]=="bf16" else torch.float16
        if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 unavailable; do not silently change precision")
        base=backend.model
        base.to(dtype=dtype)
        targets=sorted({name.split(".")[-1] for name,module in base.named_modules()
            if isinstance(module,torch.nn.Linear) and not any(x in name for x in ("vision_tower","vision_resampler","mm_projector","lm_head"))})
        model=get_peft_model(base,LoraConfig(r=tc["lora_r"],lora_alpha=tc["lora_alpha"],
            lora_dropout=tc["lora_dropout"],target_modules=targets,bias="none",task_type="CAUSAL_LM"))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or any(x in n for x in ("vision_tower","mm_projector")) for n,p in trainable):
            raise RuntimeError("Only language LoRA parameters may train")
        model.config.use_cache=False

        for m in model.modules():
            if isinstance(m,torch.nn.Dropout): m.p=0.0
        if tc["gradient_checkpointing"]:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

        model.config.tokenizer_model_max_length=None
        model.train()
        ddp=DistributedDataParallel(model,device_ids=[0],find_unused_parameters=False) if world>1 else model
        optimizer=torch.optim.AdamW([p for _,p in trainable],lr=tc["learning_rate"],weight_decay=tc["weight_decay"])
        batch=tc["batch_size"]; accum=tc["gradient_accumulation_steps"]
        per_rank=math.ceil(len(rows)/world)
        batches=math.ceil(per_rank/batch)
        per_epoch=math.ceil(batches/accum); total=per_epoch*tc["epochs"]
        warmup=math.ceil(total*tc["warmup_ratio"])
        def schedule(step):
            if step<warmup: return (step+1)/max(1,warmup)
            return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
        import time
        started=time.monotonic()
        token_count=0
        start_epoch,start_group,global_step=0,0,0
        checkpoint=out/"checkpoint.pt"
        if resume:
            if not checkpoint.exists(): raise ValueError("No complete optimizer checkpoint available")
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)
            set_peft_model_state_dict(model,state["adapter"])
            optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            start_epoch,start_group,global_step=state["next_epoch"],state["next_group"],state["step"]
            token_count=state.get("tokens_per_rank",[0]*world)[rank]
            rng=state["rng"][rank]
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
        tokenizer=backend.tokenizer

        def tensors(row,answer):
            ids,pixels,_=backend._inputs(row["img_path"],row["prompt"])
            tokens=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
            full=torch.cat([ids,torch.tensor([tokens],device=ids.device,dtype=ids.dtype)],dim=1)
            expanded=full.shape[1]+model.get_vision_tower().num_patches-1
            if expanded>tc["max_length"]:
                raise ContextOverflow(f"Training pair {row['image_id']} expands to {expanded}; limit {tc['max_length']}. No pair dropped/truncated")
            labels=torch.full_like(full,-100); labels[:,ids.shape[1]:]=full[:,ids.shape[1]:]
            with torch.no_grad():
                inp,pos,attention,_,embeds,labels=model.prepare_inputs_labels_for_multimodal(
                    full,None,torch.ones_like(full,dtype=torch.bool),None,labels,pixels.to(dtype),[backend._size])
            if (labels!=-100).sum().item()!=len(tokens): raise RuntimeError("Completion mask alignment failed")
            return dict(input_ids=inp,position_ids=pos,attention_mask=attention,inputs_embeds=embeds,use_cache=False),labels

        for row in rows:
            for side in ("chosen","rejected"): tensors(row,row[side])
        if rank==0:
            atomic_json(out/"preflight.json",{"pairs":len(rows),"context_checked":True,"base":backend.meta,
                "trainable_parameters":sum(p.numel() for _,p in trainable),"targets":targets,
                "dropout_effective":0,"training_dtype":str(dtype),"reference_frozen":True})
        def pack_pair(chosen, cl, rejected, rl):
            length=max(cl.shape[1],rl.shape[1])
            embeds=torch.zeros((2,length,chosen["inputs_embeds"].shape[-1]),dtype=dtype,device="cuda")
            labels=torch.full((2,length),-100,dtype=cl.dtype,device="cuda")
            attention=torch.zeros((2,length),dtype=torch.bool,device="cuda")
            for i,(inputs,lab) in enumerate(((chosen,cl),(rejected,rl))):
                n=lab.shape[1]
                embeds[i,:n]=inputs["inputs_embeds"][0]
                labels[i,:n]=lab[0]
                attention[i,:n]=True
            return {"inputs_embeds":embeds,"attention_mask":attention,"use_cache":False},labels
        def logp(call,inputs,labels):
            logits=call(**inputs,return_dict=True).logits[:,:-1].float()
            target=labels[:,1:]; mask=target!=-100
            safe=target.masked_fill(~mask,0)
            return (torch.log_softmax(logits,dim=-1).gather(-1,safe.unsqueeze(-1)).squeeze(-1)*mask).sum(dim=-1)
        def save(next_epoch,next_group):
            rng={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state()}
            all_rng=[None]*world
            all_tokens=[None]*world
            if world>1:
                dist.all_gather_object(all_rng,rng)
                dist.all_gather_object(all_tokens,token_count)
            else: all_rng=[rng]; all_tokens=[token_count]
            if rank==0:
                state={"adapter":{k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},
                    "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                    "next_epoch":next_epoch,"next_group":next_group,"step":global_step,"rng":all_rng,"tokens_per_rank":all_tokens}
                temp=out/"checkpoint.tmp"; torch.save(state,temp); os.replace(temp,checkpoint)
            if world>1: dist.barrier()
        for epoch in range(start_epoch,tc["epochs"]):
            rng=random.Random(tc["seed"]+epoch)
            indices=list(range(len(rows))); rng.shuffle(indices)

            total_samples=per_rank*world
            indices=(indices*math.ceil(total_samples/len(indices)))[:total_samples][rank::world]
            micro=[indices[i:i+batch] for i in range(0,len(indices),batch)]
            groups=[micro[i:i+accum] for i in range(0,len(micro),accum)]
            for gi,group in enumerate(groups):
                if epoch==start_epoch and gi<start_group: continue
                optimizer.zero_grad(set_to_none=True)
                count=sum(map(len,group)); mean_loss=0.0
                for minibatch in group:
                    for idx in minibatch:
                        row=rows[idx]
                        chosen,cl=tensors(row,row["chosen"]); rejected,rl=tensors(row,row["rejected"])
                        paired,labels=pack_pair(chosen,cl,rejected,rl)
                        token_count+=int((labels!=-100).sum())
                        with torch.autocast("cuda",dtype=dtype):
                            with torch.no_grad(), model.disable_adapter():
                                ref_logps=logp(model,paired,labels)
                                ref=ref_logps[0]-ref_logps[1]
                            policy_logps=logp(ddp,paired,labels)
                            policy=policy_logps[0]-policy_logps[1]
                            loss=-torch.nn.functional.logsigmoid(tc["beta"]*(policy-ref))
                        if not torch.isfinite(loss): raise RuntimeError("Nonfinite DPO loss")
                        mean_loss+=float(loss.detach())/count
                        scaler.scale(loss/count).backward()
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],tc["max_grad_norm"],error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); scheduler.step(); global_step+=1
                if rank==0:
                    print(f"DPO epoch={epoch+1} step={global_step}/{total} loss={mean_loss:.6f}",flush=True)
                    with (out/"training.jsonl").open("a") as f:
                        import json
                        f.write(json.dumps({"step":global_step,"epoch":epoch,"loss_rank0":mean_loss,"grad_norm":float(grad_norm),"lr":scheduler.get_last_lr()[0]})+"\n")
                next_epoch,next_group=(epoch+1,0) if gi==len(groups)-1 else (epoch,gi+1)
                if global_step%tc["save_every_steps"]==0 or gi==len(groups)-1: save(next_epoch,next_group)
        if rank==0:
            adapter=out/"adapter"; model.save_pretrained(adapter,safe_serialization=True)
            model.config.use_cache=True; model.config.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)

            torch.save({},adapter/"non_lora_trainables.bin")
            atomic_json(out/"complete.json",{"status":"complete","steps":global_step,"adapter":str(adapter),
                "adapter_sha256":sha256(adapter/"adapter_model.safetensors"),"elapsed_seconds_this_launch":time.monotonic()-started,"completion_tokens_rank0":token_count,
                "distributed_padding_per_epoch":per_rank*world-len(rows)})
        if world>1: dist.barrier(); dist.destroy_process_group()
