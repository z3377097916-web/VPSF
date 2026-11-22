#!/usr/bin/env python3
"""VPSF-V Stage 2 (M2 + M3): Span-Level Repair, Multi-Slot Foil-Calibrated Verification, and Dual-Route Pair Construction.

Strictly implements Sections 3.3-3.4 and Algorithm 1 (Lines 10-18) of the paper:
  - M2 (Span-Level Repair & Calibrated Verification):
      1. Localized edits: or-alternative resolution, speculative hedge-tail trimming, explicit absence-filler
         removal, and coreference-safe sentence surgery preserving original paragraph boundaries.
      2. Span-aligned token uncertainty recomputation U_pos^K(c') (Eq. 10) via tokenizer offsets.
      3. Multi-slot entity verification across subject, spatial/relational object, and identity predicate nominals
         using WordNet singular lemmatization (Eq. 11).
      4. Image-specific foil false-support calibration (rho_m, Eqs. 12-13) for debiasing continuous fact precision
         and tightening singleton verification floors.
  - M3 (Dual-Route Preference Construction):
      1. Route A (Risk-Anchored Positive y_A^+, Eqs. 14-17): conservative precision floor anchored by frozen
         13-feature logistic risk scorer R(c) within margin +0.022.
      2. Route B (Coverage-Oriented Positive y_B^+, Eq. 18): distinct positive maximizing multiplicative
         precision-coverage utility over remaining admitted candidates.
      3. Length- and paragraph-structure-aligned hard negative selection y_r^- (Eqs. 19-20) using frozen
         50-feature negative scorer with asymmetric length and paragraph-mismatch penalties.
      4. Pair deduplication and per-image balanced gradient weighting w_k (Eq. 21).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VPSF-V Step 2 (M2 + M3): Repair, multi-slot verification, and dual-route preference pair construction"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "work/llava_v15_7b",
        help="Working directory containing Stage 1 verification outputs.",
    )
    parser.add_argument(
        "--base-run",
        type=str,
        default="stage1_explore",
        help="Stage 1 exploration run name.",
    )
    parser.add_argument(
        "--guided-run",
        type=str,
        default="stage1b_clean_guided",
        help="Stage 1B clean guided regeneration run name.",
    )
    parser.add_argument(
        "--out-run",
        type=str,
        default="stage2_pairs",
        help="Output subdirectory under <work-dir>/data/ for constructed preference pairs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="VPSF_V",
        help="Experiment tag for pair manifest and training config.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CPU worker processes for deterministic pair construction.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'generation'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/v3_build_opt4_pairs.py"),
        "--v3-root",
        str(work_dir),
        "--v2-root",
        str(work_dir),
        "--source-run",
        args.base_run,
        "--guided-run",
        args.guided_run,
        "--out-run",
        args.out_run,
        "--run-group",
        args.tag,
        "--workers",
        str(args.workers),
    ]
    print(f"\n[VPSF-V Step 2 | M2+M3 Pair Construction] Running:\n  {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
