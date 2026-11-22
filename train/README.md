# VPSF-V Training Pipelines

This directory contains the complete, step-by-step preference construction and balanced LoRA-DPO training pipelines for **VPSF-V**, organized by backbone family:

```text
train/
├── README.md
├── llava_series/     # LLaVA-v1.5-7B & LLaVA-v1.5-13B (custom per-image weighted LoRA-DPO trainer)
└── hf_series/        # Qwen2.5-VL-3B-Instruct (serial HF backend + LLaMA-Factory LoRA-DPO)
```

## Environment Setup

### 1. LLaVA Series (`LLaVA-v1.5-7B` & `LLaVA-v1.5-13B`)

```bash
conda create -n vpsf-llava python=3.10 -y
conda activate vpsf-llava
# Install PyTorch matching your CUDA toolkit first, then:
pip install -r requirements-llava.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. HF Series (`Qwen2.5-VL-3B-Instruct` via LLaMA-Factory)

```bash
conda create -n vpsf-hf python=3.11 -y
conda activate vpsf-hf
# Install PyTorch matching your CUDA toolkit first, then:
pip install -r requirements-hf.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Quick Start

- **LLaVA-v1.5-7B / 13B**: see [`llava_series/README.md`](llava_series/README.md)
- **Qwen2.5-VL-3B-Instruct**: see [`hf_series/README.md`](hf_series/README.md)
