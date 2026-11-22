# VPSF-V LLaVA Series (`LLaVA-v1.5-7B` & `LLaVA-v1.5-13B`)

Step-by-step, paper-faithful implementation of **Algorithm 1** for `LLaVA-v1.5-7B` and `LLaVA-v1.5-13B`.

## Directory Layout

```text
train/llava_series/
├── README.md
├── step1_explore_and_verify.py      # M1: 10+(6+1)+(4+1) exploration, 11-cat extraction, dual-query VQA & foil probes
├── step2_repair_and_build_pairs.py  # M2+M3: span repair, multi-slot verification, dual-route pairing & weighting
├── step3_train_dpo_llava.sh         # Per-image balanced LoRA-DPO training (DeepSpeed ZeRO-2)
├── run_llava_pipeline.sh            # Sequential one-command runner (Step 1 -> Step 2 -> Step 3)
└── trainer/                         # Balanced LLaVA LoRA-DPO trainer + vendored LLaVA architecture
```

## Step-by-Step Execution

Prepare the unlabeled 10k image pool following [`../../data/RLAIF_dataset/README.md`](../../data/RLAIF_dataset/README.md), then run:

### Step 1 (M1): Fact-Guided Exploration & Dual-Query Self-Verification

```bash
python3 train/llava_series/step1_explore_and_verify.py \
  --work-dir work/llava_v15_7b \
  --gpus 0,1,2,3
```

### Step 2 (M2 + M3): Span-Level Repair, Multi-Slot Verification & Dual-Route Pair Construction

```bash
python3 train/llava_series/step2_repair_and_build_pairs.py \
  --work-dir work/llava_v15_7b \
  --tag VPSF_V_7B
```

### Step 3: Per-Image Balanced LoRA-DPO Training

All DPO hyperparameters strictly match RLSF-V (`LoRA r=256, alpha=512, beta=0.1, lr=4e-6, cosine, warmup=0.05, weight_decay=0.01, epochs=2, max_length=2048, global_batch=32, ZeRO-2`, frozen vision tower and projector):

```bash
# LLaVA-v1.5-7B (4 GPUs x micro 8 x accum 1 = 32)
INPUT_MODEL=liuhaotian/llava-v1.5-7b \
bash train/llava_series/step3_train_dpo_llava.sh 7b

# LLaVA-v1.5-13B (4 GPUs x micro 4 x accum 2 = 32)
INPUT_MODEL=liuhaotian/llava-v1.5-13b \
bash train/llava_series/step3_train_dpo_llava.sh 13b
```

### One-Command Sequential Runner

```bash
bash train/llava_series/run_llava_pipeline.sh 7b
bash train/llava_series/run_llava_pipeline.sh 13b
```
