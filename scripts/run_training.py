#!/usr/bin/env python3
"""四卡（或等价八卡）联合 LoRA-DPO 训练启动器（支持 E3-V3 多节点 step checkpoint 保存以供开发集选优）。"""
import argparse, json, sys, os, subprocess, hashlib, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preference.protocol import POLICY, GROUPS, FEEDBACK_BOUNDARY, PROMPT
from preference.selectors import SCHEMA
from preference.source_contract import validate_source, fingerprint
from preference.purity import evaluate, RULE


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(cfg, group, data, out, resume=None):
    args = [
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node', str(len(cfg['gpu_ids'])), str(ROOT / 'train/llava_series/trainer/dpo_llava.py'),
    ]
    ckpt_policy = cfg.get('checkpoint_policy', 'last_epoch_2')
    save_strategy = 'steps' if ckpt_policy == 'steps_150_plus_final' else 'epoch'
    save_limit = 6 if ckpt_policy == 'steps_150_plus_final' else 2
    settings = dict(
        deepspeed=str(ROOT / 'configs/zero2.json'), model_name_or_path=cfg['model_path'], vision_tower=cfg['vision_path'],
        output_dir=str(out), run_name=group, train_data_path=str(data), beta=cfg['beta'], learning_rate=cfg['learning_rate'],
        num_train_epochs=cfg['epochs'], per_device_train_batch_size=cfg['micro_batch'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation'], warmup_steps=0, warmup_ratio=cfg['warmup_ratio'], weight_decay=cfg['weight_decay'],
        lr_scheduler_type='cosine', lora_enable=True, lora_r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_bias='none', mm_projector_lr=0,
        freeze_mm_mlp_adapter=True, bits=16, mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2,
        image_aspect_ratio='pad', mm_use_im_start_end=False, mm_use_im_patch_token=False, conv_version='v1',
        model_max_length=cfg['max_length'], bf16=True, tf32=True, gradient_checkpointing=True, ddp_find_unused_parameters=False,
        dataloader_num_workers=cfg.get('workers', 4), report_to='none', eval_strategy='no', save_strategy=save_strategy,
        save_total_limit=save_limit, logging_steps=1, seed=cfg.get('seed', 42),
    )
    if ckpt_policy == 'steps_150_plus_final':
        settings['save_steps'] = int(cfg.get('save_steps', 150))
    if resume:
        settings['resume_from_checkpoint'] = str(resume)
    for k, v in settings.items():
        args.extend(['--' + k, str(v)])
    return args


def idle_cards(ids):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    mapping = {int(x.split(',')[0]): x.split(',')[1].strip() for x in rows.splitlines()}
    if any(i not in mapping for i in ids):
        raise RuntimeError('Requested GPU indexes not present')
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    busy = [line for line in procs.splitlines() if line.split(',')[0].strip() in {mapping[i] for i in ids}]
    if busy:
        raise RuntimeError('Requested GPUs have active compute processes; stop without waiting/killing: ' + str(busy))


APPROVED_SCHEME = {
    'learning_rate': 4e-6, 'beta': 0.1, 'lora_rank': 256, 'lora_alpha': 512,
    'max_length': 2048, 'warmup_ratio': 0.05, 'weight_decay': 0.01,
    'epochs': 2,
}
ALLOWED_CHECKPOINT_POLICIES = {'last_epoch_2', 'steps_150_plus_final'}


def validate_approved_scheme(cfg):
    for key, expected in APPROVED_SCHEME.items():
        if cfg.get(key) != expected:
            raise ValueError(f'Approved E3 scheme requires {key}={expected!r}; got {cfg.get(key)!r}')
    if cfg.get('checkpoint_policy') not in ALLOWED_CHECKPOINT_POLICIES:
        raise ValueError(f"Approved checkpoint_policy must be in {sorted(ALLOWED_CHECKPOINT_POLICIES)}; got {cfg.get('checkpoint_policy')!r}")


EQUIVALENT_PARALLEL_SCHEMES = {'four_gpu_joint_e3': 4, 'eight_gpu_joint_e3': 8}


def validate_parallel_scheme(cfg):
    mode = cfg.get('parallel_mode')
    if mode not in EQUIVALENT_PARALLEL_SCHEMES:
        raise ValueError('Use an approved joint training mode: ' + ', '.join(EQUIVALENT_PARALLEL_SCHEMES))
    ids = cfg['gpu_ids']
    expected = EQUIVALENT_PARALLEL_SCHEMES[mode]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f'Approved mode {mode} requires exactly {expected} distinct GPUs')
    accum = cfg['gradient_accumulation']
    if expected * cfg['micro_batch'] * accum != 32:
        raise ValueError('Global batch must remain 32 (8 GPUs x micro 4, or 4 GPUs x micro 8)')
    if cfg['micro_batch'] != 32 // (expected * accum):
        raise ValueError('Per-device micro batch must keep the global batch at 32')
