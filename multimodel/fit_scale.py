import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_TASK_ROOT), str(_TASK_ROOT / "generation")]
"""Fit unlabeled source-candidate moments once; no evaluation data are read."""
import argparse, json, math
from pathlib import Path
from multimodel.prepare import digest, dump
from multimodel.settings import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.binding.read_text())
    R = Path(b["run_root"])
    manifest = json.loads((R / "manifests/source_index.json").read_text())
    from v2.storage import read

    vals = {k: dict(n=0, mean=0.0, m2=0.0) for k in ("u", "words")}
    sources = []
    skipped = set()
    data_root = R / "data/full10000_clean_v3"
    for sp in list(data_root.glob("skipped_shard_*.jsonl")) + list(data_root.glob("*/skipped_shard_*.jsonl")):
        for line in sp.read_text().splitlines():
            if line.strip():
                skipped.add(str(json.loads(line)["id"]))
    for sh in range(4):
        for e in manifest[sh::4]:
            rp = R / "data/full10000_clean_v3" / str(sh) / "results" / (e["id"] + ".json.gz")
            if not rp.exists():
                if str(e["id"]) in skipped:
                    continue
                raise FileNotFoundError(str(rp))
            result = read(rp)
            source = Path(result["source_file"])
            if digest(source) != result["source_sha256"]:
                raise ValueError("Source changed: " + str(source))
            raw = read(source)
            if raw["generation_identity"]["model"]["profile"] != b["profile"]:
                raise ValueError("Cross-model scale input")
            if len(raw["candidates"]) != 22:
                raise ValueError("Scale requires all 22 original candidates")
            sources.append(dict(id=e["id"], sha256=result["source_sha256"]))
            for c in raw["candidates"]:
                for k, x in [("u", c["fuzzy"]["u"]), ("words", len(c["text"].split()))]:
                    if x is None or not math.isfinite(float(x)):
                        raise ValueError("Invalid scale feature")
                    q = vals[k]
                    q["n"] += 1
                    delta = float(x) - q["mean"]
                    q["mean"] += delta / q["n"]
                    q["m2"] += delta * (float(x) - q["mean"])
    if len(sources) < 0.98 * len(manifest):
        raise ValueError("Too many skipped images for scale: %d/%d" % (len(sources), len(manifest)))
    ref = json.loads((ROOT / "models/results_v1_2.json").read_text())["risk_model"]
    for k, q in vals.items():
        q["std"] = math.sqrt(q.pop("m2") / q["n"])
        ix = ref["features"].index(k)
        if q["std"] < 1e-12:
            raise ValueError("Zero source feature variance")
        q.update(reference_mean=ref["mu"][ix], reference_std=ref["sd"][ix])
    scale = R / "adaptation/feature_scale.json"
    if scale.exists():
        raise FileExistsError("Scale is frozen; use a new adaptation run to change it")
    dump(
        scale,
        dict(
            model_lock_sha256=b["model_lock_sha256"],
            features=vals,
            source_images=len(sources),
            sources=sources,
            variance="population",
            reference="full precision values from results_v1_2.json",
        ),
    )
    b["scale_path"] = str(scale)
    b["scale_sha256"] = digest(scale)
    dump(R / "binding_opt4.json", b)
    print(scale)


if __name__ == "__main__":
    main()
