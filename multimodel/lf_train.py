import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_TASK_ROOT), str(_TASK_ROOT / "generation")]
"""LLaMA-Factory worker; keep image preprocessing limits consistent with construction."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--binding", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.binding.read_text())
    cfg = json.loads(a.config.read_text())
    from multimodel.prepare import digest

    identity = json.loads((a.config.parent / "export_identity.json").read_text())
    if (
        digest(a.config) != identity["config_sha256"]
        or digest(a.config.parent / "vpsf_v_opt4_dpo.json")
        != identity["dataset_sha256"]
    ):
        raise ValueError("Exported training inputs changed")
    from multimodel.settings import verified_inventory

    if digest(b["model_lock"]) != b["model_lock_sha256"]:
        raise ValueError("Model lock changed after preparation")
    verified_inventory(b["model_lock"])
    import llamafactory.model.loader as loader

    original = loader.patch_processor

    def patch(processor, tokenizer, model_args):
        original(processor, tokenizer, model_args)
        if b["profile"]["family"] == "internvl":
            if not hasattr(processor.image_processor, "max_patches"):
                raise RuntimeError("InternVL HF image processor incompatible")
            processor.image_processor.max_patches = 6
            processor.image_processor.crop_to_patches = True
            # LF v0.9.4 passes max_patches=12 explicitly; override at the called boundary.
            image_processor = processor.image_processor
            original_preprocess = image_processor.preprocess

            def preprocess(*args, **kwargs):
                kwargs["max_patches"] = 6
                return original_preprocess(*args, **kwargs)

            image_processor.preprocess = preprocess
            # Native HF tiling sees the source image; do not add LF's separate area downscale.
            processor.image_max_pixels = 2**63 - 1
            processor.image_min_pixels = 1

    loader.patch_processor = patch
    from llamafactory.train.tuner import run_exp

    run_exp(cfg)


if __name__ == "__main__":
    main()
