#!/usr/bin/env bash
# VPSF-V Stage 3: Per-Image Balanced LoRA-DPO Training for LLaVA-v1.5-7B / 13B
# Matches paper Section 4.1 & Appendix B.1:
#   LoRA r=256, alpha=512, DPO beta=0.1, lr=4e-6, cosine decay (warmup=0.05, wd=0.01),
#   epochs=2, max_length=2048, global_batch=32, DeepSpeed ZeRO-2, frozen vision tower & projector.
set -euo pipefail

SIZE="${1:-7b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${SIZE}" == "7b" ]]; then
  MODEL_PATH="${INPUT_MODEL:-liuhaotian/llava-v1.5-7b}"
  MICRO_BATCH="${MICRO_BATCH:-8}"
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
elif [[ "${SIZE}" == "13b" ]]; then
  MODEL_PATH="${INPUT_MODEL:-liuhaotian/llava-v1.5-13b}"
  MICRO_BATCH="${MICRO_BATCH:-4}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
else
  echo "Usage: bash step3_train_dpo_llava.sh [7b|13b]" >&2
  exit 1
fi

VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${REPO_ROOT}/work/llava_v15_${SIZE}/data/stage2_pairs/training_clean/pairs/E3_V3_OPT4.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-${SIZE}-vpsf-v}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo "${GPUS}" | awk -F',' '{print NF}')"

echo "[VPSF-V Step 3] Training LLaVA-v1.5-${SIZE} LoRA-DPO"
echo "  MODEL_PATH      = ${MODEL_PATH}"
echo "  VISION_TOWER    = ${VISION_TOWER}"
echo "  TRAIN_DATA_PATH = ${TRAIN_DATA_PATH}"
echo "  OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "  GPUS            = ${GPUS} (nproc=${NUM_GPUS}, micro=${MICRO_BATCH}, accum=${GRAD_ACCUM}, global=$((NUM_GPUS * MICRO_BATCH * GRAD_ACCUM)))"

CUDA_VISIBLE_DEVICES="${GPUS}" \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/llava_series/trainer:${PYTHONPATH:-}" \
python3 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/trainer/dpo_llava.py" \
  --deepspeed "${REPO_ROOT}/configs/zero2.json" \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "vpsf-v-llava-${SIZE}" \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --beta 0.1 \
  --learning_rate 4e-6 \
  --num_train_epochs 2 \
  --per_device_train_batch_size "${MICRO_BATCH}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --lora_enable True \
  --lora_r 256 \
  --lora_alpha 512 \
  --lora_bias none \
  --mm_projector_lr 0 \
  --freeze_mm_mlp_adapter True \
  --bits 16 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --image_aspect_ratio pad \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --conv_version v1 \
  --model_max_length 2048 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --dataloader_num_workers 4 \
  --report_to none \
  --eval_strategy no \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 1 \
  --seed 42
