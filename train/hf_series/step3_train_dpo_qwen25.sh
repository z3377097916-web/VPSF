#!/usr/bin/env bash
# VPSF-V Stage 3: LoRA-DPO Training for Qwen2.5-VL-3B-Instruct via LLaMA-Factory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONFIG_YAML="${1:-${SCRIPT_DIR}/qwen25_vl_3b_dpo.yaml}"

echo "[VPSF-V HF Step 3] Launching LLaMA-Factory LoRA-DPO training"
echo "  GPUS        = ${GPUS}"
echo "  CONFIG_YAML = ${CONFIG_YAML}"

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES="${GPUS}" \
  llamafactory-cli train "${CONFIG_YAML}"
