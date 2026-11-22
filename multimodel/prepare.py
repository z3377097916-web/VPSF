import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_TASK_ROOT), str(_TASK_ROOT / "generation")]
"""Materialize an isolated model run and immutable checkpoint inventory."""
import argparse, hashlib, json, shutil, sys, os, tempfile
from pathlib import Path
from multimodel.settings import PROFILES, ROOT


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, p)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def download(repo, revision, dest, group, legacy=False):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    patterns = ["*.json", "*.model", "*.jinja", "*.txt", "*.py"] + (
        ["*.bin"] if legacy else ["*.safetensors"]
    )
    snapshot_download(
        repo, revision=info.sha, local_dir=str(dest), allow_patterns=patterns
    )
    entries = []
    records = []
    for item in info.siblings:
        p = dest / item.rfilename
        if not p.is_file():
            continue
        actual = digest(p)
        lfs = getattr(item, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs else None
        if expected and actual != expected:
            raise ValueError("Hub SHA256 mismatch: " + str(p))
        entries.append(dict(name=item.rfilename, official_sha256=actual))
        records.append(
            dict(
                group=group,
                name=item.rfilename,
                revision=info.sha,
                sha256=actual,
                path=str(p.resolve()),
            )
        )
    if not any(r["name"].endswith((".bin", ".safetensors")) for r in records):
        raise ValueError("No model weights downloaded")
    return dict(repo=repo, sha=info.sha, files=entries), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Existing clean 10k run containing manifests, checks and heldout image inventory",
    )
    ap.add_argument("--revision", default="main")
    ap.add_argument("--vision-revision", default="main")
    a = ap.parse_args()
    R = a.run_root.resolve()
    data = a.data_root.resolve()
    spec = PROFILES[a.profile]
    if R.exists() and any(R.iterdir()):
        raise FileExistsError("Use a new isolated run directory: " + str(R))
    R.mkdir(parents=True, exist_ok=True)
    entries = json.loads((data / "manifests/source_index.json").read_text())
    if len(entries) != 10000 or len({x["id"] for x in entries}) != 10000:
        raise ValueError("Need unique clean 10k manifest")
    for e in entries:
        if not Path(e["image"]).is_absolute():
            e["image"] = str((data / e["image"]).resolve())
        e["requires_generation"] = True
        e.pop("source", None)
        e.pop("source_sha256", None)
    dump(R / "manifests/source_index.json", entries)
    for rel in [
        "evaluation/heldout_image_manifest_all.json",
        "checks/training_image_pixel_hashes.json",
    ]:
        dst = R / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / rel, dst)
    dump(R / "manifests/training_images.json", entries)
    # Private code snapshot prevents two concurrent model runs from changing each other's model manifest.
    code = R / "code"
    shutil.copytree(ROOT, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    models = R / "models"
    manifest = {}
    records = []
    manifest["base"], rec = download(
        spec["repo"], a.revision, models / "base", "base", spec["family"] == "llava"
    )
    records += rec
    if spec["family"] == "llava":
        manifest["vision"], rec = download(
            "openai/clip-vit-large-patch14-336",
            a.vision_revision,
            models / "vision",
            "vision",
            True,
        )
        records += rec
    dump(code / "runtime/model_manifest.json", manifest)
    dump(
        models / "verified_files.json",
        [{k: r[k] for k in ("group", "name", "revision", "sha256")} for r in records],
    )
    lock = R / "model_lock.json"
    dump(lock, dict(profile=a.profile, repositories=manifest, files=records))
    cfg = json.loads((code / "configs/generation.json").read_text())
    cfg["fuzzy"]["top_r"] = spec["top_r"]
    cfg["generation"]["max_new_tokens"] = spec.get("max_new_tokens", 512)
    cfg["batching"]["max_batch"] = spec.get("vqa_max_batch", 32)
    cfg.update(
        models=str(models),
        vendor=str(code / "runtime/vendor"),
        backend="llava" if spec["family"] == "llava" else "hf",
        context_len=spec["context_len"],
    )
    dump(R / "evaluation/generation.json", cfg)
    dump(R / "construction_generation.json", cfg)
    dump(code / "configs/generation.json", cfg)
    b = dict(
        profile_name=a.profile,
        profile=spec,
        run_root=str(R),
        model_dir=str(models / "base"),
        models_root=str(models),
        vendor=str(code / "runtime/vendor"),
        model_lock=str(lock),
        model_lock_sha256=digest(lock),
        generation_config=str(R / "construction_generation.json"),
        scale_path=None,
        caption_context_len=spec["context_len"],
    )
    dump(R / "binding.json", b)
    # Preparation keeps the binding immutable during candidate generation. A second binding adds fitted scale later.
    train = json.loads((code / "configs/run.json").read_text())
    from v2.repair import POLICY

    train.update(
        backbone=a.profile,
        data_policy=POLICY,
        model_path=str(models / "base"),
        vision_path=str(models / "vision"),
        output_root=str(R / "training/models"),
        evaluation_manifest=str(R / "evaluation/heldout_image_manifest_all.json"),
        training_image_manifest=str(R / "manifests/training_images.json"),
        micro_batch=spec["micro_batch"],
        gradient_accumulation=spec["gradient_accumulation"],
        learning_rate=spec["learning_rate"],
    )
    dump(R / "training_handoff/a100_training.json", train)
    dump(
        R / "preparation.json",
        dict(
            input_data_root=str(data),
            manifest_sha256=digest(R / "manifests/source_index.json"),
            model_lock_sha256=digest(lock),
        ),
    )
    print(
        "Prepared:",
        R,
        "\nNext: python",
        code / "multimodel/run.py",
        "--binding",
        R / "binding.json",
        "--stage construct",
    )


if __name__ == "__main__":
    main()
