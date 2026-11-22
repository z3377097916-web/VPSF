import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_TASK_ROOT), str(_TASK_ROOT / "generation")]
"""User-requested repeated-row LLaMA-Factory export with audit trail."""
import argparse, json
from pathlib import Path
from multimodel.prepare import dump, digest
from multimodel.settings import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.binding.read_text())
    R = Path(b["run_root"])
    p = b["profile"]
    from scripts.v3_export_llamafactory_dpo import export_llamafactory

    out = R / "training/lf"
    out.mkdir(parents=True, exist_ok=True)
    export_llamafactory(a.pairs, out)
    tag = {
        "qwen25_3b": "qwen2_5_vl_3b",
        "qwen3_8b": "qwen3_vl_8b",
        "internvl35_8b": "internvl3_5_8b",
    }[b["profile_name"]]
    import yaml

    cfg = yaml.safe_load((out / f"train_dpo_{tag}.yaml").read_text())
    cfg.update(
        model_name_or_path=b["model_dir"],
        deepspeed=str(ROOT / "configs/zero2.json"),
        output_dir=str(R / "training/models/E3_V3_OPT4"),
        per_device_train_batch_size=p["micro_batch"],
        gradient_accumulation_steps=p["gradient_accumulation"],
        learning_rate=p["learning_rate"],
        cutoff_len={"qwen25_3b": 2560, "qwen3_8b": 2560, "internvl35_8b": 3072}[b["profile_name"]],
        enable_thinking=False,
        overwrite_output_dir=False,
        report_to="none",
    )
    if "max_pixels" in p:
        cfg["image_max_pixels"] = p["max_pixels"]
    if p["family"] == "internvl":
        cfg.update(crop_to_patches=True, default_system="")
        ds = out / "vpsf_v_opt4_dpo.json"
        records = json.loads(ds.read_text())
        for record in records:
            record["conversations"][0]["value"] = record["conversations"][0][
                "value"
            ].replace("<image>", "<image>\n", 1)
        dump(ds, records)
    if 4 * int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"]) != 32:
        raise ValueError("Global batch must be 32 (paper main.tex:378)")
    dump(out / "train.json", cfg)
    dump(
        out / "export_identity.json",
        dict(
            source_pairs_sha256=digest(a.pairs),
            model_lock_sha256=b["model_lock_sha256"],
            config_sha256=digest(out / "train.json"),
            dataset_sha256=digest(out / "vpsf_v_opt4_dpo.json"),
            weighting="Repeat singleton pairs twice, retain each of two pairs once. Equal image occurrence counts; different epoch/update schedule from pair_weight training.",
        ),
    )
    print(out / "train.json")


if __name__ == "__main__":
    main()
