#!/usr/bin/env python3
"""VPSF-V Stage 1 (M1): Fact-Guided Exploration & Dual-Query Self-Verification (Qwen2.5-VL-3B-Instruct).

Uses the canonical serial HuggingFace backend (`multimodel.hf_backend.SerialHFVisionBackend`)
without batching or multi-machine speed approximations:
  - Native AutoProcessor image preprocessing and chat template (`max_pixels = 1,003,520`).
  - Fast BPE tokenizer offset mapping for POS-aligned top-R fuzzy uncertainty (R = 20, max_new_tokens = 512).
  - Plain-prose normalization of Markdown emphasis and bulleted lists prior to spaCy dependency parsing.
  - Multi-round 10 + (6+1) + (4+1) exploration, dual-query Yes/No self-VQA, foil probes, and clean guided regeneration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n[VPSF-V HF Step 1 | {label}] Running:\n  {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VPSF-V HF Step 1 (M1): Serial Qwen2.5-VL-3B-Instruct exploration & dual-query self-verification"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local HuggingFace directory for Qwen2.5-VL-3B-Instruct.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "work/qwen25_vl_3b",
        help="Working directory for Qwen2.5-VL-3B-Instruct construction outputs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/RLAIF_dataset/rlaif10k.jsonl",
        help="Path to the 10,000-image unlabeled JSONL manifest.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'generation'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    # 1. Prepare canonical run binding & model profile (R=20, max_new_tokens=512)
    binding_path = work_dir / "binding.json"
    if not binding_path.exists():
        profile = json.loads((REPO_ROOT / "configs/profiles/qwen25_vl_3b.json").read_text(encoding="utf-8"))
        binding = {
            "profile_name": "qwen25_3b",
            "profile": profile,
            "model_dir": str(args.model_dir.resolve()),
            "run_root": str(work_dir),
            "manifest": str(args.manifest.resolve()),
        }
        binding_path.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")

    # 2. Launch Stage 1A (10 + (6+1) + (4+1) exploration & dual-query VQA)
    run_cmd(
        "M1-A: Base Multi-Round Exploration & Dual-Query VQA (Qwen2.5-VL-3B)",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v2_launch.py"),
            "--v2-root",
            str(work_dir),
            "--run-name",
            "full10000_clean_v3",
            "--gpus",
            args.gpus,
        ],
        env,
    )

    # 3. Launch Stage 1B (clean double-supported guided regeneration on bottleneck images)
    run_cmd(
        "M1-B1: Select Bottleneck Images",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3p_regen_clean_guided.py"),
            "select-bottlenecks",
            "--v3-root",
            str(work_dir),
            "--base-run",
            "full10000_clean_v3",
            "--out-run",
            "full10000_v3p_guided",
        ],
        env,
    )
    run_cmd(
        "M1-B2: Clean Double-Supported Guided Regeneration",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3p_regen_clean_guided.py"),
            "launch",
            "--v3-root",
            str(work_dir),
            "--v2-root",
            str(work_dir),
            "--base-run",
            "full10000_clean_v3",
            "--out-run",
            "full10000_v3p_guided",
            "--gpus",
            args.gpus,
        ],
        env,
    )


if __name__ == "__main__":
    main()
