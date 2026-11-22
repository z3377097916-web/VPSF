#!/usr/bin/env python3
"""将 VPSF-V (E3_V3_OPT4) 标准偏好对 JSON 导出为 LLaMA-Factory 多模态 DPO 训练格式与各架构标准 YAML 配置。

严格对齐 RLSF-V (ICML 2026, Appendix B.1) 跨架构训练超参数：
- Qwen2.5-VL-3B-Instruct: r=256, alpha=512, beta=0.1, lr=4e-6, cosine, warmup=0.05, wd=0.01, epochs=2, max_len=2048
- Qwen3-VL-8B-Instruct:   r=256, alpha=512, beta=0.1, lr=5e-6, cosine, warmup=0.05, wd=0.01, epochs=2, max_len=2048, freeze_vision_tower=true, freeze_multi_modal_projector=true
- InternVL3.5-8B-HF:      r=256, alpha=512, beta=0.1, lr=5e-6, cosine, warmup=0.05, wd=0.01, epochs=2, max_len=2048, freeze_vision_tower=true, freeze_multi_modal_projector=true

注意按图等权（Per-Image Equal Weight）精确保持：
- 在 VPSF-V 中，单图 1 对的样本相对权重为单图 2 对样本的 2 倍（使每张源图总梯度权重恒等）。
- LLaMA-Factory 原生 DPO DataLoader 按样本等权采样，因此本导出器默认对单图仅 1 对的样本精确复制 2 次、单图 2 对的样本各保留 1 次，使每张源图在每个 Epoch 中严格贡献完全相等的 2 次梯度更新（per-image weight spread == 0.0）！
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


YAML_SPECS = {
    "qwen2_5_vl_3b": {
        "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "template": "qwen2_vl",
        "learning_rate": "4.0e-6",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
    "qwen3_vl_8b": {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "template": "qwen2_vl",
        "learning_rate": "5.0e-6",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
    },
    "internvl3_5_8b": {
        "model_name_or_path": "OpenGVLab/InternVL3_5-8B-HF",
        "template": "intern_vl",
        "learning_rate": "5.0e-6",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
    },
}


def export_llamafactory(pairs_path: Path, out_dir: Path, image_root: Path | None = None):
    rows = json.loads(pairs_path.read_text(encoding="utf-8"))
    counts = collections.Counter(str(r["image_id"]) for r in rows)
    if max(counts.values(), default=0) > 2:
        raise ValueError(f"Expected <= 2 pairs per image, got max={max(counts.values())}")

    out_dir.mkdir(parents=True, exist_ok=True)
    lf_rows = []
    for r in rows:
        iid = str(r["image_id"])
        img_p = str(r["img_path"])
        if image_root is not None and not Path(img_p).is_absolute():
            img_p = str((image_root / img_p).resolve())
        prompt_txt = r["prompt"].replace("<image>", "").strip()
        rec = {
            "conversations": [{"from": "human", "value": f"<image>{prompt_txt}"}],
            "chosen": {"from": "gpt", "value": r["chosen"]},
            "rejected": {"from": "gpt", "value": r["rejected"]},
            "images": [img_p],
        }
        repeats = 2 if counts[iid] == 1 else 1
        for _ in range(repeats):
            lf_rows.append(rec)

    ds_name = "vpsf_v_opt4_dpo"
    data_file = out_dir / f"{ds_name}.json"
    data_file.write_text(json.dumps(lf_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ds_info = {
        ds_name: {
            "file_name": data_file.name,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "images": "images",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(ds_info, ensure_ascii=False, indent=2), encoding="utf-8")

    for tag, sp in YAML_SPECS.items():
        yaml_txt = f"""### Auto-generated LLaMA-Factory DPO config for VPSF-V ({tag})
model_name_or_path: {sp['model_name_or_path']}
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 256
lora_alpha: 512
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid
freeze_vision_tower: true
freeze_multi_modal_projector: true
deepspeed: examples/deepspeed/ds_z2_config.json

### dataset
dataset_dir: {out_dir.resolve()}
dataset: {ds_name}
template: {sp['template']}
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: {out_dir.resolve() / ('checkpoints_' + tag)}
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train (4 GPUs -> global batch = {4 * sp['per_device_train_batch_size'] * sp['gradient_accumulation_steps']})
per_device_train_batch_size: {sp['per_device_train_batch_size']}
gradient_accumulation_steps: {sp['gradient_accumulation_steps']}
learning_rate: {sp['learning_rate']}
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
ddp_timeout: 180000000
seed: 42
"""
        (out_dir / f"train_dpo_{tag}.yaml").write_text(yaml_txt, encoding="utf-8")

    summary = {
        "source_pairs": len(rows),
        "unique_images": len(counts),
        "exported_equal_weight_records": len(lf_rows),
        "records_per_image": 2,
        "dataset_file": str(data_file),
        "yamls": [str(out_dir / f"train_dpo_{k}.yaml") for k in YAML_SPECS],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="VPSF-V (E3_V3_OPT4) pairs JSON 路径")
    ap.add_argument("--out-dir", type=Path, required=True, help="导出的 LLaMA-Factory 数据集与 YAML 目录")
    ap.add_argument("--image-root", type=Path, default=None, help="若 img_path 为相对路径，指定图像根目录")
    a = ap.parse_args()
    export_llamafactory(a.pairs, a.out_dir, a.image_root)


if __name__ == "__main__":
    main()
