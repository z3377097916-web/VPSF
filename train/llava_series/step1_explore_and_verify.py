#!/usr/bin/env python3
"""VPSF-V Stage 1 (M1): Fact-Guided Candidate Exploration & Dual-Query Self-Verification (LLaVA-v1.5-7B/13B).

Strictly implements Section 3.2 and Algorithm 1 (Lines 2-9) of the paper:
  1. Multi-round candidate sampling under the canonical 10 + (6+1) + (4+1) schedule:
     - Stage 0: 10 unprompted stochastic samples under X_0 ("Describe this image."), caching logits for POS-filtered
                top-R fuzzy uncertainty U_pos(c) (Eq. 4).
     - Stage 1 & 2: Two-step neutralized checklists (Entity Existence -> Conditional Attributes/Counts/Relations)
                generating 6+1 and 4+1 samples, reserving 1 unprompted X_0 sample per round for unbiased recurrence.
  2. 11-category atomic proposition extraction & dual-query Yes/No self-VQA verification (Eqs. 5-6) plus
     image-specific synthetic foil probes to estimate affirmative-bias rate rho_m.
  3. Clean guided bottleneck regeneration (Algorithm 1, Lines 7-9): for images lacking >=48-word clean candidates,
     regenerate guided descriptions conditioned strictly on double-supported entities and re-verify with the base MLLM.

No external annotators, no external teacher models, and no training-speed approximations are used.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n[VPSF-V Step 1 | {label}] Running:\n  {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VPSF-V Step 1 (M1): Multi-round fact-guided exploration & dual-query self-verification (LLaVA)"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "work/llava_v15_7b",
        help="Working directory containing manifests/source_index.json and saving Stage 1 results.",
    )
    parser.add_argument(
        "--base-run",
        type=str,
        default="stage1_explore",
        help="Run name for Stage 0/1/2 exploration & dual-query verification.",
    )
    parser.add_argument(
        "--guided-run",
        type=str,
        default="stage1b_clean_guided",
        help="Run name for clean double-supported guided bottleneck regeneration.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs (one worker per GPU).",
    )
    parser.add_argument(
        "--skip-clean-guided",
        action="store_true",
        help="Skip Stage-1B bottleneck clean guided regeneration (only run base 10+(6+1)+(4+1) exploration).",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'generation'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    # Step 1A: Multi-round 10 + (6+1) + (4+1) exploration, 11-cat extraction, dual-query VQA, and foil probes
    run_cmd(
        "M1-A: Base Multi-Round Exploration & Dual-Query VQA",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v2_launch.py"),
            "--v2-root",
            str(work_dir),
            "--run-name",
            args.base_run,
            "--gpus",
            args.gpus,
        ],
        env,
    )

    if args.skip_clean_guided:
        return

    # Step 1B: Identify bottleneck images lacking >=48-word verified candidates and regenerate cleanly
    run_cmd(
        "M1-B1: Select Bottleneck Images for Clean Guided Regeneration",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3p_regen_clean_guided.py"),
            "select-bottlenecks",
            "--v3-root",
            str(work_dir),
            "--base-run",
            args.base_run,
            "--out-run",
            args.guided_run,
        ],
        env,
    )

    run_cmd(
        "M1-B2: Run Clean Double-Supported Guided Regeneration & Re-Verification",
        [
            sys.executable,
            str(REPO_ROOT / "scripts/v3p_regen_clean_guided.py"),
            "launch",
            "--v3-root",
            str(work_dir),
            "--v2-root",
            str(work_dir),
            "--base-run",
            args.base_run,
            "--out-run",
            args.guided_run,
            "--gpus",
            args.gpus,
        ],
        env,
    )


if __name__ == "__main__":
    main()
