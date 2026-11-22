# VPSF-V: Coverage-Preserving Hallucination Mitigation in MLLMs via Calibrated Proposition Self-Feedback

Official repository for **VPSF-V** (*Verified Proposition Self-Feedback*), a self-contained, reference-policy alignment framework that achieves state-of-the-art multimodal hallucination suppression while overcoming the *omission shortcut* of uncertainty-only self-feedback—using **zero human preference labels** and **zero external teacher models**.

---

## Overview

Existing alignment pipelines for mitigating visual hallucinations in MLLMs face a sharp trade-off:
- **RLHF / RLAIF** (e.g., human annotations, GPT-4V, Gemini Vision, DeepSeek-V3, external detectors) is costly to scale and shifts preference data off-policy.
- **Uncertainty-only self-feedback** (e.g., RLSF-V) eliminates external supervision, but minimizing whole-response top-$R$ token uncertainty induces an **omission shortcut** that lowers hallucination rates largely by leaving visible scene objects and relations unexpressed.

**VPSF-V** shifts reference-policy self-feedback from whole-response uncertainty to **calibrated proposition-level visual verification** via three sequential stages using *only the base MLLM*:

1. **M1 — Fact-Guided Candidate Exploration (`generation/rlsfv/`)**: Compiles previously supported claims into neutralized two-step checklists (*Entity Existence* $\to$ *Conditional Attributes/Counts/Relations*) under a $10 + (6+1) + (4+1)$ sampling schedule, while reserving unprompted draws for unbiased recurrence estimation and performing clean double-supported regeneration on bottleneck images.
2. **M2 — Span-Level Repair & Foil-Calibrated Multi-Slot Verification (`v2/`)**: Excises localized flaws (`or`-alternatives, speculative noun guesses, explicit absence fillers, and coreference-safe flawed sentences) while recomputing span-aligned token uncertainty $U_{\mathrm{pos}}^{\mathcal{K}}$, enforces WordNet-lemmatized verification across **all concrete entity slots** (subject, relational object, predicate nominal), and calibrates singleton admission via image-specific synthetic foil probes ($\rho_m$).
3. **M3 — Dual-Route Preference Construction & Balanced LoRA-DPO (`preference/`, `train/`)**: Decouples candidate admission from selection by pairing complementary **risk-anchored** ($y_A^+$) and **coverage-oriented** ($y_B^+$) positives with length- and paragraph-structure-aligned hard negatives ($y_r^-$) under per-image balanced LoRA-DPO.

---

## Main Results

Under a unified deterministic evaluation protocol (`temperature = 0`, zero external LLM judges) with a `10k`-image unlabeled pool and **zero external supervision**, **VPSF-V** achieves the lowest generative hallucination rates across all 12 compared RLHF, RLAIF, detector-supervised, and self-feedback baselines on all three evaluated backbones, while simultaneously recovering substantial object coverage and relational understanding over RLSF-V:

| Backbone & Method | Feedback | AMBER CHAIR↓ | AMBER Cover↑ | AMBER HalRate↓ | AMBER Cog↓ | ObjHal Resp↓ | ObjHal Ment↓ | AMBER Dis Acc↑ | AMBER Dis F1↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLaVA-v1.5-7B** (Base) | ✗ | 7.7 | 51.7 | 35.5 | 4.3 | 49.33 | 14.94 | 71.6 | 74.1 |
| +OPA-DPO | GPT-4V | 2.6 | 45.5 | 12.2 | 0.9 | 17.67 | 5.47 | 81.3 | 85.5 |
| +SENTINEL | Detectors | 2.9 | 43.8 | 14.8 | 1.3 | 9.00 | 2.57 | 76.1 | 79.3 |
| +symMPO | DeepSeek-V3 | 5.9 | 50.5 | 28.2 | 3.6 | 26.67 | 7.22 | 81.4 | **86.7** |
| +RLSF-V | Self-Feedback | 2.4 | 33.3 | 5.1 | 0.3 | 5.67 | 2.89 | 78.8 | 84.0 |
| **+VPSF-V (Ours)** | **Self-Feedback** | **1.4** | **40.5** | **4.6** | **0.1** | **3.67** | **1.07** | **81.9** | 85.7 |
| **LLaVA-v1.5-13B** (Base) | ✗ | 6.8 | 52.0 | 31.5 | 3.4 | 48.33 | 13.86 | 71.3 | 73.1 |
| +RLSF-V | Self-Feedback | 2.0 | 33.5 | 4.4 | 0.3 | 2.67 | 1.44 | **84.8** | 87.6 |
| **+VPSF-V (Ours)** | **Self-Feedback** | **1.1** | **36.7** | **3.3** | **0.2** | **2.33** | **1.42** | 78.6 | 81.5 |
| **Qwen2.5-VL-3B-Instruct** (Base) | ✗ | 8.0 | 69.7 | 51.2 | 5.5 | 27.33 | 7.76 | 88.0 | 91.0 |
| +RLSF-V | Self-Feedback | 7.2 | 66.5 | 42.3 | 4.8 | 25.67 | 6.29 | **88.2** | **91.1** |
| **+VPSF-V (Ours)** | **Self-Feedback** | **1.1** | 45.2 | **4.7** | **0.3** | **14.33** | **4.47** | 85.9 | 88.8 |

> On **LLaVA-v1.5-7B**, VPSF-V also achieves the lowest MSCOCO-500 hallucination rates (`CHAIR_s = 16.40%`, `CHAIR_i = 9.01%`) while expanding caption length `2.45×–2.89×` over RLSF-V (`61.7` vs. `25.2` words) and improving AMBER Relation F1 by **`+12.0` points** (`60.4% → 72.4%`).

---

## Repository Structure

This repository is organized for maximum clarity and step-by-step faithfulness to **Algorithm 1** and **Sections 3.2–3.4** of the paper (no opaque multi-machine sharding or training-speed approximations):

```text
VPSF-V/
├── LICENSE
├── README.md
├── requirements-llava.txt             # Dependencies for LLaVA-v1.5-7B / 13B
├── requirements-hf.txt                # Dependencies for Qwen2.5-VL-3B (HF + LLaMA-Factory)
├── configs/                           # Paper hyperparameters, ZeRO-2 config, and model profiles
│   ├── generation.json
│   ├── run.json
│   ├── zero2.json
│   └── profiles/                      # llava_v15_7b.json, llava_v15_13b.json, qwen25_vl_3b.json
├── data/
│   └── RLAIF_dataset/                 # 10k unlabeled image manifest & zero-overlap provenance
│       ├── README.md
│       ├── rlaif10k.jsonl
│       ├── replenished_provenance.json
│       └── training_image_isolation.json
├── models/                            # Frozen calibrated M3 scoring parameters (Eqs. 14-15, 20)
│   ├── formula_origin.json
│   ├── results_v1_2.json              # 13-feature logistic risk scorer & anchor weights
│   └── hc_reject_models.json          # 50-feature Route-A / Route-B hard-negative scorers
├── generation/                        # M1: Multi-round exploration, 11-cat extraction & dual-query VQA
│   ├── configs/
│   └── rlsfv/
├── v2/                                # M2: Span-level repair, coreference-safe surgery & ledger guard
├── preference/                        # M3: Purity checks, frozen scorers & per-image weighting
├── multimodel/                        # Clean serial HF backend & Eq. (23) moment standardization
├── scripts/                           # Canonical stage implementations called by train/ entry points
├── train/
│   ├── README.md
│   ├── llava_series/                  # Step-by-step LLaVA-v1.5-7B/13B pipeline + balanced DPO trainer
│   │   ├── step1_explore_and_verify.py
│   │   ├── step2_repair_and_build_pairs.py
│   │   ├── step3_train_dpo_llava.sh
│   │   ├── run_llava_pipeline.sh
│   │   └── trainer/
│   └── hf_series/                     # Step-by-step Qwen2.5-VL-3B pipeline + LLaMA-Factory configs
│       ├── step1_explore_and_verify.py
│       ├── step2_build_and_export_lf.py
│       ├── step3_train_dpo_qwen25.sh
│       ├── qwen25_vl_3b_dpo.yaml
│       └── run_qwen25_pipeline.sh
├── eval/                              # Unified deterministic 5-benchmark evaluation suite
│   ├── README.md
│   ├── eval_five_benchmarks.py
│   ├── eval_amber_gen.py
│   └── eval_amber_disc.py
└── tests/                             # Contract, foil-calibration & zero-leakage unit tests
```

---

## Quick Start

### 1. Prepare Unlabeled Images

Download images from [`openbmb/RLAIF-V-Dataset`](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) and place them under `data/RLAIF_dataset/images/`. See [`data/RLAIF_dataset/README.md`](data/RLAIF_dataset/README.md) for details.

### 2. LLaVA-v1.5-7B & LLaVA-v1.5-13B

Install the environment (`train/README.md`), then run the three stages sequentially:

```bash
# Run all three stages (M1 Exploration -> M2+M3 Repair & Pairing -> Balanced LoRA-DPO)
bash train/llava_series/run_llava_pipeline.sh 7b
bash train/llava_series/run_llava_pipeline.sh 13b
```

Or run each step individually (`step1_explore_and_verify.py` → `step2_repair_and_build_pairs.py` → `step3_train_dpo_llava.sh`). See [`train/llava_series/README.md`](train/llava_series/README.md).

### 3. Qwen2.5-VL-3B-Instruct (HF + LLaMA-Factory)

```bash
MODEL_DIR=/path/to/Qwen2.5-VL-3B-Instruct \
bash train/hf_series/run_qwen25_pipeline.sh
```

See [`train/hf_series/README.md`](train/hf_series/README.md) for step-by-step execution and LLaMA-Factory training configuration.

### 4. Evaluation & Unit Tests

```bash
# Run contract & zero-leakage unit tests
python3 -m unittest discover -s tests -p "test_v*.py" -v

# Evaluate on AMBER, Object HalBench, MSCOCO-500, POPE, and ScienceQA
python3 eval/eval_five_benchmarks.py --help
```

---

## Acknowledgements

Our LLaVA-v1.5 DPO trainer and unlabeled 10k image pool build upon [RLSF-V](https://github.com/XLearning-SCU/RLSF-V), [LLaVA](https://github.com/haotian-liu/LLaVA), [RLAIF-V](https://github.com/RLHF-V/RLAIF-V), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the authors for releasing their code and datasets.

---

## Citation

If you find **VPSF-V** useful in your research, please consider citing our paper:

```bibtex
@article{zhang2027vpsfv,
  title={VPSF-V: Coverage-Preserving Hallucination Mitigation in MLLMs via Calibrated Proposition Self-Feedback},
  author={Zhang, Xuanyuan and Li, Yongxiang},
  year={2027}
}
```
