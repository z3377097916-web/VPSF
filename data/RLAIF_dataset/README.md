# RLAIF-V Unlabeled Image Pool (`10,000` Images)

This directory mirrors the unlabeled image pool used in [RLSF-V](https://github.com/XLearning-SCU/RLSF-V):

```text
data/RLAIF_dataset/
├── README.md
├── rlaif10k.jsonl                  # 10,000-image manifest (same pool as RLSF-V)
├── replenished_provenance.json     # Provenance of 6 overlap-free replacement images
├── training_image_isolation.json   # SHA-256 & decoded RGB pixel-hash zero-overlap audit
└── images/                         # Place downloaded RLAIF-V images here
```

## Preparing Images

1. Download the image files from the Hugging Face dataset [`openbmb/RLAIF-V-Dataset`](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset).
2. Place the extracted images into:

```text
data/RLAIF_dataset/images/
```

3. Ensure each `image_path` entry in `rlaif10k.jsonl` resolves to the corresponding local image file before launching Step 1.

## Strict Zero-Overlap Guarantee

Both file-level SHA-256 checksums and decoded RGB pixel hashes were verified against all five evaluation benchmarks (`AMBER`, `MSCOCO-500`, `Object HalBench`, `POPE`, and `ScienceQA`). Six historical overlapping images detected during auditing were replaced with non-overlapping images from the same `openbmb/RLAIF-V-Dataset` source pool (`replenished_provenance.json`, `training_image_isolation.json`). No external annotations, human labels, or preference judgments from `RLAIF-V-Dataset` are ever read—only the raw images are used.
