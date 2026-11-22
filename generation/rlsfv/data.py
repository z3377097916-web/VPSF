from pathlib import Path
from .io import records, sha256, read_json, atomic_json, fingerprint
from .config import X0
from .scoring import score, guard_refuted, verified
from .facts import norm

def manifest(path, official_amber_vendor=None):
    root = Path(path).resolve().parent
    rows = records(path)
    seen, hashes = set(), {}
    official = None
    if official_amber_vendor is not None:
        from .amber import queries
        official = {q['id']: q['image'] for q in queries(official_amber_vendor, 'a')}
    def official_test(row):
        return (official is not None and row.get('source') == 'AMBER'
                and row['split'] == 'test' and type(row['id']) is int
                and official.get(row['id']) == Path(row['image']).name)
    for r in rows:


        r.setdefault("split", "train")
        if set(r) - {"id", "image", "split", "source"} or not {"id", "image", "split"} <= r.keys():
            raise ValueError("Manifest fields: id, image, optional split (default train), optional source")
        if type(r["id"]) not in (str, int) or str(r["id"]) in seen:
            raise ValueError("Invalid or duplicate image ID")
        seen.add(str(r["id"]))
        if r["split"] not in ("train", "dev", "test", "diagnostic"):
            raise ValueError("Unknown split")
        r["image"] = str((root / r["image"]).resolve())
        h = sha256(r["image"])
        if h in hashes and not (official_test(r) and official_test(hashes[h])):
            raise ValueError(f"Exact duplicate image: {hashes[h]['id']} and {r['id']}; resolve explicitly")
        hashes[h] = r
    if not rows:
        raise ValueError("Empty manifest")
    return rows

def leakage_check(paths, output):
    inventories, seen = [], {}
    for path in paths:
        for r in manifest(path):
            h = sha256(r["image"])
            if h in seen:
                raise ValueError(f"Cross-manifest duplicate {r['id']} / {seen[h]}")
            seen[h] = {"id": r["id"], "split": r["split"]}
            inventories.append(dict(r, sha256=h))
    report = {"exact_duplicates": 0, "images": inventories,
              "near_duplicate_status": "NOT_AUDITED", "note": "Exact hashes cannot exclude crops/re-encodings. Independently audit near duplicates before formal training."}
    atomic_json(output, report)
    return report

def results(run):
    run = Path(run)
    status = read_json(run / "status.json")
    if not status["complete"]:
        raise ValueError("Run contains pending/failed images; cannot silently export a partial dataset")
    out = [read_json(p) for p in sorted((run / "images").glob("*/result.json"))]
    if len(out) != status["total"] or any(r["status"] != "complete" for r in out):
        raise ValueError("Result inventory mismatch")
    return out

def pair_type(result, chosen, rejected):
    a, b = set(chosen["claim_ids"]), set(rejected["claim_ids"])
    f = result["facts"]
    rel = lambda k: f[k]["category"] in {"spatial", "relation"} and f[k]["modality"] == "definite"
    adds = [k for k in a-b if rel(k)]
    removes = [k for k in b-a if rel(k)]
    rule = result.get("ranking", {}).get("guard_rule", "any_valid_refutation")
    rule = "double_refuted" if rule == "double_refutation" else "any_refuted"
    supported = lambda k: verified(f[k]) and all(v["status"] == "supported" for v in f[k]["checks"])
    refuted = lambda k: guard_refuted(f[k], rule)
    correction, completion = any(refuted(k) for k in removes), any(supported(k) for k in adds)
    if correction and completion:
        return "mixed"
    if correction:
        return "correction"
    if completion:
        return "completion"
    return "unknown" if adds or removes else "no_relation"

def export(run, output, scoring=None, seed=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new export directory")
    rows = results(run)
    if scoring is not None:
        for r in rows:
            r["ranking"] = score(r["candidates"], r["facts"], scoring, seed, r.get("foil_calibration"))
    methods = list(rows[0]["ranking"]["selections"])
    datasets, captions, audits = {}, {}, {}
    for method in methods:
        pairs, answers, audit = [], [], []
        for r in rows:
            s = r["ranking"]["selections"][method]
            by_id = {c["id"]: c for c in r["candidates"]}
            c = by_id[s["chosen"]]
            answers.append({"id": r["record"]["id"], "response": c["text"]})
            entry = dict(s, image_id=r["record"]["id"], split=r["record"]["split"],
                         chosen_words=len(c["text"].split()), category_counts={})
            selected_row = next(x for x in r["ranking"]["rows"] if x["id"] == c["id"])
            entry.update(guard_rule=s.get("guard_rule", r["ranking"].get("guard_rule", "legacy_double_refutation")
                                          if method.startswith("G-") else "unguarded"),
                         primary_coverage=selected_row.get("coverage_components", {}),
                         auxiliary_coverage=selected_row.get("auxiliary_coverage", {}),
                         category_coverage_audit_only=selected_row.get("categories", {}),
                         chosen_error_ids=selected_row.get("errors", []),
                         verified_precision_proxy=selected_row.get("verified_precision"),
                         primary_qF=selected_row.get("coverage", {}).get("qF"),
                         legacy_coverage_qv=selected_row.get("legacy_coverage_qv"),
                         legacy_qF=selected_row.get("legacy_qF"))
            if s["rejected"]:
                d = by_id[s["rejected"]]
                entry.update(rejected_words=len(d["text"].split()), relation_pair_type_proxy=pair_type(r,c,d))
                if s["pair_valid"] and r["record"]["split"] == "train":
                    pairs.append({"image_id": r["record"]["id"], "img_path": r["record"]["image"],
                        "image_sha256": r["image_sha256"], "prompt": X0, "chosen": c["text"], "rejected": d["text"],
                        "chosen_id": c["id"], "rejected_id": d["id"], "split": "train"})
            audit.append(entry)
        datasets[method], captions[method], audits[method] = pairs, answers, audit
    for method in methods:
        atomic_json(output / f"{method}.pairs.json", datasets[method])
        atomic_json(output / f"{method}.captions.json", captions[method])
        atomic_json(output / f"{method}.audit.json", audits[method])
        if method.startswith("G-"):
            parent = method[2:]
            ids = {p["image_id"] for p in datasets[method]} & {p["image_id"] for p in datasets[parent]}
            atomic_json(output / f"{method}.matched.pairs.json", [p for p in datasets[method] if p["image_id"] in ids])
            atomic_json(output / f"{parent}.matched-to-{method}.pairs.json", [p for p in datasets[parent] if p["image_id"] in ids])

    training_methods = [m for m in methods if m not in ("RAND", "RLSF10")]
    shared = set.intersection(*({p["image_id"] for p in datasets[m]} for m in training_methods))

    shared_order = [r["record"]["id"] for r in rows if r["record"]["id"] in shared]
    for m in training_methods:
        lookup = {p["image_id"]: p for p in datasets[m]}
        atomic_json(output / f"{m}.common10.pairs.json", [lookup[i] for i in shared_order])
    atomic_json(output / "common10.images.json", shared_order)
    def median(v):
        v = sorted(v)
        return (v[(len(v)-1)//2] + v[len(v)//2]) / 2 if v else None
    def stage_counts(m):
        out = {}
        for x in audits[m]:
            st = x.get("chosen_stage")
            out[str(st)] = out.get(str(st), 0) + 1
        return out
    def mean_or_none(values):
        values = [x for x in values if x is not None]
        return sum(values) / len(values) if values else None
    summary = {m: {"chosen_stage_counts": stage_counts(m),
                  "median_chosen_over_rejected_words": median([x["chosen_words"] / x["rejected_words"] for x in audits[m]
                                                               if x.get("rejected_words")]),
                  "length_matched_ablation_advised": (median([x["chosen_words"] / x["rejected_words"] for x in audits[m]
                                                              if x.get("rejected_words")]) or 1) > 1.3,
                  "train_pairs": len(datasets[m]), "captions": len(captions[m]),
                  "fallback_count": sum(x["fallback"] for x in audits[m]),
                  "fallback_rate": sum(x["fallback"] for x in audits[m])/len(rows),
                  "mean_primary_qF": mean_or_none(x.get("primary_qF") for x in audits[m]),
                  "mean_legacy_qF": mean_or_none(x.get("legacy_qF") for x in audits[m]),
                  "skipped_pairs": sum(not x["pair_valid"] for x in audits[m])} for m in methods}

    diagnostic_rows = []
    for m in methods:
        for a in audits[m]:
            diagnostic_rows.append({"method": m, "image_id": a["image_id"],
                                    "fallback": a["fallback"], "guard_rule": a["guard_rule"],
                                    "chosen_error_ids": a["chosen_error_ids"],
                                    "primary_coverage": a["primary_coverage"],
                                    "primary_qF": a.get("primary_qF"),
                                    "legacy_coverage_qv": a.get("legacy_coverage_qv"),
                                    "legacy_qF": a.get("legacy_qF"),
                                    "auxiliary_coverage": a["auxiliary_coverage"],
                                    "category_coverage_audit_only": a["category_coverage_audit_only"]})
    atomic_json(output / "coverage-proxy-diagnostics.json", {
        "warning": "Internal model-supported proxies, not COCO/AMBER ground-truth coverage or relation metrics.",
        "rows": diagnostic_rows})
    run_identity = read_json(Path(run) / "run.json")
    atomic_json(output / "summary.json", {"source_run": str(Path(run).resolve()), "source_hash": sha256(Path(run)/"run.json"),
        "protocol_revision": rows[0]["ranking"].get("protocol_revision"),
        "guard_rule": rows[0]["ranking"].get("guard_rule"),
        "model": run_identity.get("models"), "run_identity": run_identity,
        "scoring": scoring or rows[0]["ranking"]["config"], "methods": summary, "common10_size": len(shared_order), "common10_empty_requires_decision": not shared_order,
        "warning": "Caption-selection outputs measure generative performance only. Cross-model absolute CHAIR/Cover comparisons are NOT valid; compare within model or as relative deltas only."})
    return summary

def validate_pairs(path):
    rows = records(path)
    seen = set()
    for p in rows:
        if p.get("split") != "train" or p.get("prompt") != X0:
            raise ValueError("DPO accepts train-split pairs under common x0 only")
        if p["image_id"] in seen or norm(p["chosen"]) == norm(p["rejected"]) or not p["chosen"].strip() or not p["rejected"].strip():
            raise ValueError("Duplicate image or invalid preference pair")
        if sha256(p["img_path"]) != p["image_sha256"]:
            raise ValueError("Training image changed")
        seen.add(p["image_id"])
    if not rows:
        raise ValueError("No valid training pairs")
    return rows
