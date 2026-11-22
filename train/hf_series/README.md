# VPSF-V HuggingFace Series (`Qwen2.5-VL-3B-Instruct`)

Step-by-step, paper-faithful implementation of **VPSF-V** for HuggingFace vision-language backbones (`Qwen2.5-VL-3B-Instruct`) using the canonical serial HF backend (`multimodel/hf_backend.py`) and LLaMA-Factory LoRA-DPO.

## Directory Layout

```text
train/hf_series/
├── README.md
├── step1_explore_and_verify.py     # M1: Serial HF backend exploration & dual-query VQA (R=20, max_new_tokens=512)
├── step2_build_and_export_lf.py    # M2+M3: Eq. (23) moment standardization, pairing & balanced LLaMA-Factory export
├── step3_train_dpo_qwen25.sh       # LLaMA-Factory 4-GPU ZeRO-2 LoRA-DPO launcher
├── qwen25_vl_3b_dpo.yaml           # Exact LLaMA-Factory DPO configuration (r=256, alpha=512, beta=0.1, lr=4e-6)
├── dataset_info.json               # LLaMA-Factory ShareGPT ranking dataset registration
└── run_qwen25_pipeline.sh          # Sequential one-command runner (Step 1 -> Step 2 -> Step 3)
```

## Step-by-Step Execution

### Step 1 (M1): Fact-Guided Exploration & Dual-Query Self-Verification

```bash
python3 train/hf_series/step1_explore_and_verify.py \
  --model-dir /path/to/Qwen2.5-VL-3B-Instruct \
  --work-dir work/qwen25_vl_3b \
  --gpus 0,1,2,3
```

### Step 2 (M2 + M3): Unlabeled Moment Standardization, Pair Construction & Balanced Export

Computes unlabeled backbone feature moments $(\mu_{b,x}, s_{b,x})$ (Eq. 23), constructs deduplicated dual-route pairs, and exports them to LLaMA-Factory format while duplicating single-pair images once so every active image contributes equal gradient mass (Eq. 21):

```bash
python3 train/hf_series/step2_build_and_export_lf.py \
  --work-dir work/qwen25_vl_3b \
  --image-root data/RLAIF_dataset
```

### Step 3: LoRA-DPO Training with LLaMA-Factory

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train/hf_series/step3_train_dpo_qwen25.sh train/hf_series/qwen25_vl_3b_dpo.yaml
```
