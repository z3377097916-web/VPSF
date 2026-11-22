#!/usr/bin/env bash
# End-to-End VPSF-V Pipeline for Qwen2.5-VL-3B-Instruct (Step 1 -> Step 2 -> Step 3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:?Please set MODEL_DIR=/path/to/Qwen2.5-VL-3B-Instruct}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work/qwen25_vl_3b}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== [VPSF-V Qwen2.5-VL-3B] Step 1/3: Fact-Guided Exploration & Dual-Query Verification (M1) ==="
python3 "${SCRIPT_DIR}/step1_explore_and_verify.py" \
  --model-dir "${MODEL_DIR}" \
  --work-dir "${WORK_DIR}" \
  --gpus "${GPUS}"

echo "=== [VPSF-V Qwen2.5-VL-3B] Step 2/3: Moment Standardization, M2+M3 Pairing & LLaMA-Factory Export ==="
python3 "${SCRIPT_DIR}/step2_build_and_export_lf.py" \
  --work-dir "${WORK_DIR}"

echo "=== [VPSF-V Qwen2.5-VL-3B] Step 3/3: Balanced LoRA-DPO Training via LLaMA-Factory ==="
CUDA_VISIBLE_DEVICES="${GPUS}" \
bash "${SCRIPT_DIR}/step3_train_dpo_qwen25.sh" "${SCRIPT_DIR}/qwen25_vl_3b_dpo.yaml"
