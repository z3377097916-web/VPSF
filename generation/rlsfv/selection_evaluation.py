"""All selection groups: official AMBER + independent caption-relation assessment.

This module never generates captions, trains, or uses self-verification as ground truth.
Missing relation annotations remain null/not_evaluated, never a zero or a shared QA score.
"""
import csv
import re
from pathlib import Path
from . import amber, diagnostics
from .io import atomic_json, read_json, records, fingerprint, sha256, code_hash
from .scoring import PRIMARY_GROUP


def groups(selection, greedy=None):
    root = Path(selection)
    methods = read_json(root / "summary.json")["methods"]
    names = list(methods)
    if not names or len(set(names)) != len(names) or PRIMARY_GROUP not in names:
        raise ValueError("Selection summary must include unique groups and the primary " + PRIMARY_GROUP)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Unsafe/invalid method name in selection summary")
    paths = {n: root / (n + ".captions.json") for n in names}
    if greedy is not None:
        paths = {"greedy": Path(greedy), **paths}
    ids = None
    for name, path in paths.items():
        rows = records(path)
        keys = [str(x["id"]) for x in rows]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Empty/duplicate caption IDs: " + name)
        if any(not isinstance(x.get("response"), str) or not x["response"].strip() for x in rows):
            raise ValueError("Invalid caption text: " + name)
        if ids is not None and set(keys) != ids:
            raise ValueError("All methods must have identical image IDs: " + name)
        ids = set(keys)
    return paths


def _csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def relation_groups(selection, output, annotation_index=None, greedy=None, repetitions=1000, seed=42):
    paths = groups(selection, greedy)
    root = Path(output)
    index = {}
    if annotation_index:
        source = Path(annotation_index)
        index = read_json(source)["methods"]
        if not isinstance(index, dict) or set(index) - paths.keys():
            raise ValueError("Relation index contains unknown methods")
        index = {k: (source.parent / v).resolve() for k, v in index.items()}
    rows, reports, reference = [], {}, None
    for name, path in paths.items():
        row = {"method": name, "status": "not_evaluated", "reason": "independent_annotations_missing"}
        for cat in diagnostics.REL_CATEGORIES + ("spatial_and_relation",):
            for metric in ("precision", "recall", "f1"):
                row[cat + "." + metric] = None
        if name in index:
            annotation = read_json(index[name])
            expected = {str(x["id"]): x["response"] for x in records(path)}
            images = annotation["images"]
            actual = {str(x["id"]): x["caption"] for x in images}
            if len(actual) != len(images) or actual != expected or annotation.get("method") != name:
                raise ValueError("Relation annotations do not match selected captions/group: " + name)
            if any(x.get("extraction_missed") for x in images):
                raise ValueError("Incomplete relation extraction must be resolved before scoring: " + name)

            ref = fingerprint(sorted([{"id": str(x["id"]), "references": sorted(x["references"], key=lambda r: str(r["id"]))}
                                      for x in images], key=lambda x: x["id"]))
            if reference is not None and ref != reference:
                raise ValueError("Groups used different independent reference inventories")
            reference = ref
            result = diagnostics.relation_metrics(index[name], root / (name + ".json"), repetitions, seed)
            result["status"] = "complete"
            atomic_json(root / (name + ".json"), result)
            reports[name] = result
            row.update(status="complete", reason=None)
            for cat in diagnostics.REL_CATEGORIES + ("spatial_and_relation",):
                for metric in ("precision", "recall", "f1"):
                    row[cat + "." + metric] = result["metrics"][cat][metric]
        rows.append(row)
    report = {"status": "complete" if all(r["status"] == "complete" for r in rows) else "not_fully_evaluated",
              "reference_inventory_sha256": reference, "metrics_scale": "fraction_0_to_1",
              "truth": "independent_annotations_only", "results": rows, "reports": reports,
              "note": "Official shared relation QA is NOT caption-relation assessment. Missing values are not zero."}
    atomic_json(root / "summary.json", report)
    _csv(root / "summary.csv", rows)
    return report


def _evaluate_cached(cfg, path, output, task):
    report_path = output / "report.json"
    if output.exists():
        if not report_path.exists():
            raise ValueError("Incomplete evaluation directory: " + str(output))
        report = read_json(report_path)
        if report.get("status") != "complete" or report.get("task") != task or report.get("input_sha256") != sha256(path):
            raise ValueError("Stale/incomplete evaluation; choose a new output directory: " + str(output))
        source = Path(cfg["vendor"]) / "AMBER/inference.py"
        assets = amber.args_for(cfg["vendor"], path, task)
        if report.get("source_sha256") != sha256(source) or report.get("official_assets") != {
            key: sha256(getattr(assets, key)) for key in ("annotation", "word_association", "safe_words", "metrics")}:
            raise ValueError("Official evaluator/assets changed; choose a new output directory")
        return report
    return amber.evaluate(path, output, cfg["vendor"], task)


def evaluate_groups(cfg, selection, output, discriminative=None, greedy=None,
                    annotation_index=None, repetitions=1000, seed=42):
    paths, root = groups(selection, greedy), Path(output)
    task = "a" if discriminative else "g"
    identity = {"code": code_hash(), "config": cfg, "task": task,
                "captions": {n: sha256(p) for n, p in paths.items()},
                "discriminative": sha256(discriminative) if discriminative else None}
    if (root / "identity.json").exists() and read_json(root / "identity.json") != identity:
        raise ValueError("Selection evaluation identity changed; use a new output directory")
    atomic_json(root / "identity.json", identity)
    rows, generation, discrimination = [], [], []
    shared_metrics = None
    for name, path in paths.items():
        if discriminative:
            answer = root / "combined" / (name + ".json")
            atomic_json(answer, amber.combine_frozen_answers(path, discriminative, cfg["vendor"]))
        else:
            answer = path
        report = _evaluate_cached(cfg, answer, root / ("eval-" + name), task)
        metrics = report.get("metrics_percent", {})
        if report.get("status") != "complete" or report.get("n") != (15220 if task == "a" else 1004) or len(metrics) != (32 if task == "a" else 4):
            raise ValueError("Incomplete official metrics: " + name)
        gen = {k: v for k, v in metrics.items() if k.startswith("generative.")}
        disc = {k: v for k, v in metrics.items() if not k.startswith("generative.")}
        if len(gen) != 4 or len(disc) != (28 if task == "a" else 0):
            raise ValueError("Official generative/discriminative inventory mismatch")
        if shared_metrics is not None and disc != shared_metrics:
            raise ValueError("Identical frozen QA produced inconsistent discriminative metrics")
        shared_metrics = disc
        rows.append(dict(method=name, **metrics))
        generation.append(dict(method=name, **gen))
        discrimination.append(dict(method=name, **disc))
    relations = relation_groups(selection, root / "caption-relations", annotation_index, greedy, repetitions, seed)
    for row, rel in zip(rows, relations["results"]):
        row["caption_relation_status"] = rel["status"]
        for metric in ("precision", "recall", "f1"):
            row["caption_relation." + metric] = rel["spatial_and_relation." + metric]
    result = {"status": "complete", "experiment_status": "development_only", "primary_group": PRIMARY_GROUP,
              "official_task": task, "official_metrics_count": 32 if task == "a" else 4,
              "model_parameters_changed": False, "discriminative_answers_shared": bool(discriminative),
              "per_method_official_relation_improvement_tested": False,
              "caption_relation_status": relations["status"], "results": rows,
              "note": "Official metrics are percentages; caption-relation metrics are fractions. No scientific success is inferred."}
    atomic_json(root / "all-groups-results.json", result)
    _csv(root / "all-groups-results.csv", rows)
    _csv(root / "generative-results.csv", generation)
    if discriminative:
        _csv(root / "shared-discriminative-results.csv", discrimination)
    return result
