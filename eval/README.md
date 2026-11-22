# VPSF-V Evaluation Suite

This directory provides our unified, deterministic (`temperature = 0`, greedy decoding, zero commercial LLM judge calls) evaluation pipeline covering all five benchmarks reported in the paper:

- **AMBER Generative** (`1,004` images): `CHAIR↓`, `Cover↑`, `HalRate↓`, `Cog↓`
- **AMBER Discriminative** (`14,216` probes across 6 aspects: *Existence, Attribute, State, Number, Action, Relation*): `Acc↑`, `F1↑`, and per-aspect `F1↑`
- **MSCOCO-500** (`500` validation images): `CHAIR_s↓`, `CHAIR_i↓`, `Cover↑`, `Obj/C`, `Words`
- **Object HalBench** (`300` instances, deterministic rule-based COCO-synonym evaluator): `Resp↓`, `Ment↓`, `Obj/C`, `Words`
- **POPE** (`9,000` items across *Random / Popular / Adversarial*): `Rand↑`, `Pop↑`, `Adv↑`, `Avg↑`
- **ScienceQA** (`4,241` multiple-choice questions): `NAT`, `SOC`, `LAN`, `TXT`, `IMG`, `NO`, `G1–6`, `G7–12`, `Overall`

## Files

```text
eval/
├── README.md
├── eval_five_benchmarks.py    # Unified 5-benchmark evaluator + summary table generator
├── eval_amber_gen.py          # Standalone AMBER Generative inference & scoring
└── eval_amber_disc.py         # Standalone AMBER Discriminative inference & 6-aspect F1 scoring
```

## Usage

### 1. Run AMBER Generative & Discriminative

```bash
python3 eval/eval_amber_gen.py --help
python3 eval/eval_amber_disc.py --help
```

### 2. Run All Five Benchmarks Under the Unified Deterministic Protocol

```bash
python3 eval/eval_five_benchmarks.py --help
```

> **Note on Inference Overhead:** Checklists, span-level repair operators, dual-query VQA verification, and candidate admission filters are **offline preference-construction tools only** and are never invoked during evaluation or test-time inference. All checkpoints are evaluated using standard single-pass greedy decoding (`do_sample=False`).
