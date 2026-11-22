#!/usr/bin/env bash
# End-to-End VPSF-V Pipeline for LLaVA-v1.5-7B / 13B (Step 1 -> Step 2 -> Step 3)
set -euo pipefail

SIZE="${1:-7b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work/llava_v15_${SIZE}}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== [VPSF-V LLaVA-${SIZE}] Step 1/3: Fact-Guided Exploration & Dual-Query Self-Verification (M1) ==="
python3 "${SCRIPT_DIR}/step1_explore_and_verify.py" \
  --work-dir "${WORK_DIR}" \
  --gpus "${GPUS}"

echo "=== [VPSF-V LLaVA-${SIZE}] Step 2/3: Span Repair, Multi-Slot Verification & Dual-Route Pairing (M2+M3) ==="
python3 "${SCRIPT_DIR}/step2_repair_and_build_pairs.py" \
  --work-dir "${WORK_DIR}" \
  --tag "VPSF_V_${SIZE}"

echo "=== [VPSF-V LLaVA-${SIZE}] Step 3/3: Per-Image Balanced LoRA-DPO Training ==="
TRAIN_DATA_PATH="${WORK_DIR}/data/stage2_pairs/training_clean/pairs/E3_V3_OPT4.json" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
bash "${SCRIPT_DIR}/step3_train_dpo_llava.sh" "${SIZE}"
