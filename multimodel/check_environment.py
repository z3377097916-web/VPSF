"""Check API-critical pins; capture the actual runtime before model work."""

import importlib.metadata as md, json, sys
from pathlib import Path


def main():
    expected = {
        "transformers": "4.57.1",
        "peft": "0.17.1",
        "trl": "0.24.0",
        "accelerate": "1.11.0",
        "spacy": "3.8.0",
        "lm-format-enforcer": "0.11.3",
    }
    actual = {k: md.version(k) for k in expected}
    bad = {k: (v, actual[k]) for k, v in expected.items() if actual[k] != v}
    if bad:
        raise RuntimeError("Wrong HF environment: " + str(bad))
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        InternVLForConditionalGeneration,
        AutoProcessor,
    )
    from llamafactory.data.template import TEMPLATES

    if not {"qwen2_vl", "qwen3_vl_nothink", "intern_vl"} <= TEMPLATES.keys():
        raise RuntimeError("LLaMA-Factory template support incomplete")
    import spacy, torch

    for n in ("en_core_web_sm", "en_core_web_lg"):
        spacy.load(n)
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    print(
        json.dumps(
            dict(
                python=sys.version,
                packages=actual,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                available_gpus=torch.cuda.device_count(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
