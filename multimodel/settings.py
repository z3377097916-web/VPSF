"""Explicit per-run binding. Never infer a backbone from a checkpoint filename."""

import functools, json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "llava13b": dict(
        repo="liuhaotian/llava-v1.5-13b",
        family="llava",
        top_r=30,
        learning_rate=4e-6,
        micro_batch=4,
        gradient_accumulation=2,
        context_len=4096,
        max_new_tokens=384,
        vqa_max_batch=32,
    ),
    "qwen25_3b": dict(
        repo="Qwen/Qwen2.5-VL-3B-Instruct",
        family="qwen2_5_vl",
        top_r=20,
        learning_rate=4e-6,
        micro_batch=1,
        gradient_accumulation=8,
        context_len=4096,
        max_new_tokens=512,
        vqa_max_batch=32,
        max_pixels=1003520,
    ),
    "qwen3_8b": dict(
        repo="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen3_vl",
        top_r=5,
        learning_rate=5e-6,
        micro_batch=1,
        gradient_accumulation=8,
        context_len=4096,
        max_new_tokens=512,
        vqa_max_batch=16,
        max_pixels=1003520,
        enable_thinking=False,
    ),
    "internvl35_8b": dict(
        repo="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
        top_r=40,
        learning_rate=5e-6,
        micro_batch=1,
        gradient_accumulation=8,
        context_len=4096,
        max_new_tokens=576,
        vqa_max_batch=8,
        max_patches=6,
    ),
}


@functools.lru_cache(maxsize=8)
def _read_binding(path):
    return json.loads(Path(path).read_text())


def binding():
    p = os.environ.get("VPSFV_BINDING")
    return _read_binding(p) if p else None


def modern():
    b = binding()
    return bool(b and b["profile"]["family"] != "llava")


def cut_limit(original_words, fixed):
    return (
        max(fixed, int((0.32 if fixed == 40 else 0.38) * original_words))
        if modern()
        else fixed
    )


def normalize_caption(text):
    """Training candidates only. Keep raw text in the generation record."""
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    paras = []
    for para in re.split(r"\n\s*\n", text):
        lines = [
            re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", x).strip()
            for x in para.splitlines()
        ]
        paras.append(" ".join(x for x in lines if x))
    return "\n\n".join(x for x in paras if x).strip()


@functools.lru_cache(maxsize=8)
def read_scale(path):
    return json.loads(Path(path).read_text())


@functools.lru_cache(maxsize=8)
def checked_scale(path, expected_sha):
    from rlsfv.io import sha256

    if sha256(path) != expected_sha:
        raise ValueError("Frozen scale file changed")
    return read_scale(path)


def aligned_candidate(candidate):
    b = binding()
    if not modern() or not b.get("scale_path"):
        return candidate
    p = Path(b["scale_path"])
    if not p.exists():
        raise FileNotFoundError("Run fit-scale before OPT4: " + str(p))
    st = checked_scale(str(p), b["scale_sha256"])
    c = dict(candidate)
    if st["model_lock_sha256"] != b["model_lock_sha256"]:
        raise ValueError("Scale belongs to another model")
    for k in ("u", "words"):
        if c.get(k) is not None:
            v = st["features"][k]
            c[k] = (
                v["reference_mean"]
                + (float(c[k]) - v["mean"]) * v["reference_std"] / v["std"]
            )
    return c


@functools.lru_cache(maxsize=8)
def verified_inventory(lock_path):
    from rlsfv.io import sha256

    lock = json.loads(Path(lock_path).read_text())
    for r in lock["files"]:
        if sha256(r["path"]) != r["sha256"]:
            raise ValueError("Checkpoint checksum failed: " + r["path"])
    return [
        {k: r[k] for k in ("group", "name", "revision", "sha256")}
        for r in lock["files"]
    ]


def expected_bound_model():
    b = binding()
    if not b:
        return None
    lock = json.loads(Path(b["model_lock"]).read_text())
    from rlsfv.io import sha256

    if sha256(b["model_lock"]) != b["model_lock_sha256"]:
        raise ValueError("Model lock changed")
    return dict(
        backend="llava" if b["profile"]["family"] == "llava" else "hf",
        variant="base",
        files=[
            {k: r[k] for k in ("group", "name", "revision", "sha256")}
            for r in lock["files"]
        ],
    )
