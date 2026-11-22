#!/usr/bin/env python3
"""VPSF-V Stage 2 (M2 + M3) for HuggingFace Backbones (Qwen2.5-VL-3B-Instruct):
  1. Fit unlabeled feature moments (mu_{b,x}, s_{b,x}) for U_pos and response length (Eq. 23)
     via `multimodel/fit_scale.py` so frozen M3 risk and hard-negative scorers transfer without refitting.
  2. Run span-level repair (with adaptive trim budgets max(44, floor(0.38*l)) and max(40, floor(0.32*l))),
     multi-slot foil-calibrated verification, and dual-route preference pair construction.
  3. Export deduplicated preference pairs into LLaMA-Factory ShareGPT DPO format via
     `scripts/v3_export_llamafactory_dpo.py`, duplicating single-pair images once so every active image
     contributes equal gradient mass (exact per-image balance, Eq. 21).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n[VPSF-V HF Step 2 | {label}] Running:\n  {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VPSF-V HF Step 2 (M2 + M3): Feature standardization, pair construction & balanced LLaMA-Factory export"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "work/qwen25_vl_3b",
        help="Working directory containing Stage 1 verification outputs and binding.json.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPO_ROOT / "data/RLAIF_dataset",
        help="Root directory resolving relative image paths for LLaMA-Factory export.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CPU workers for deterministic pair construction.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    binding_path = work_dir / "binding.json"
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'generation'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    # 1. Fit unlabeled source-pool moments (Eq. 23) for cross-backbone standardization
    if binding_path.exists() and not (work_dir / "adaptation/feature_scale.json").exists():
        run_cmd(
            "Cross-Backbone Unlabeled Moment Standardization (Eq. 23)",
            [sys.executable, str(REPO_ROOT / "multimodel/fit_scale.py"), "--binding", str(binding_path)],
            env,
        )

    # 2. Construct M2 + M3 preference pairs
    run_cmd(
        "M2+M3 Repair, Multi-Slot Verification & Dual-Route Pair Construction",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3_build_opt4_pairs.py"),
            "--v3-root",
            str(work_dir),
            "--v2-root",
            str(work_dir),
            "--run-group",
            "VPSF_V_QWEN25_3B",
            "--workers",
            str(args.workers),
        ],
        env,
    )

    # 3. Export strictly per-image balanced LLaMA-Factory dataset & YAML
    pairs_file = work_dir / "data/full10000_opt4_pairs/training_clean/pairs/E3_V3_OPT4.json"
    lf_out = work_dir / "llamafactory_dpo"
    run_cmd(
        "Export Balanced LLaMA-Factory DPO Dataset (Eq. 21)",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3_export_llamafactory_dpo.py"),
            "--pairs",
            str(pairs_file),
            "--out-dir",
            str(lf_out),
            "--image-root",
            str(args.image_root.resolve()),
        ],
        env,
    )


if __name__ == "__main__":
    main()
