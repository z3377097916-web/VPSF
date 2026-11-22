"Development set on COCO val2014 (disjoint from AMBER) and a CHAIR evaluator."
from __future__ import annotations

import json
import random
from pathlib import Path

from .io import atomic_json, records, sha256

RES = Path(__file__).parent / "resources"
COCO_DOUBLE = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal',
               'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball',
               'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
               'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer',
               'stove top oven', 'home plate', 'train track']
ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']


class Chair:
    def __init__(self):
        import nltk
        self.nltk = nltk
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.objects, self.inverse = set(), {}
        for line in (RES / "chair_synonyms.txt").read_text().splitlines():
            syn = [s.strip() for s in line.strip().split(", ") if s.strip()]
            if not syn:
                continue
            self.objects.update(syn)
            for s in syn:
                self.inverse[s] = syn[0]
        self.double = {w: w for w in COCO_DOUBLE}
        for a in ANIMALS:
            self.double["baby " + a] = a
            self.double["adult " + a] = a
        for v in ("jet", "train"):
            self.double["passenger " + v] = v
        self.double.update({"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"})

    def singular(self, w):
        return self.lemmatizer.lemmatize(w, "n")

    def words(self, caption):
        """(coco words, canonical node words) in the caption, following the original rules."""
        words = [self.singular(w) for w in self.nltk.word_tokenize(caption.lower())]
        merged, i = [], 0
        while i < len(words):
            pair = " ".join(words[i:i + 2])
            if pair in self.double:
                merged.append(self.double[pair])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        if "toilet" in merged and "seat" in merged:
            merged = [w for w in merged if w != "seat"]
        found = [w for w in merged if w in self.objects]
        return found, [self.inverse[w] for w in found]


def coco_ground_truth(instances_path, captions_path, image_ids, chair):
    instances = json.loads(Path(instances_path).read_text())
    names = {c["id"]: c["name"] for c in instances["categories"]}
    seg = {i: set() for i in image_ids}
    for a in instances["annotations"]:
        if a["image_id"] in seg:
            seg[a["image_id"]].add(chair.inverse[names[a["category_id"]]])
    truth = {i: set(v) for i, v in seg.items()}
    for a in json.loads(Path(captions_path).read_text())["annotations"]:
        if a["image_id"] in truth:
            truth[a["image_id"]].update(chair.words(a["caption"])[1])
    return seg, truth


def make_manifest(coco_images, instances_path, amber_images, n, seed, output, download=False):
    """Random COCO val2014 images with >=1 object, excluding exact duplicates of AMBER images.
    download=True fetches only the sampled images (coco_url in the annotation file) into
    coco_images instead of requiring the full 6 GB val2014 folder."""
    instances = json.loads(Path(instances_path).read_text())
    with_objects = sorted({a["image_id"] for a in instances["annotations"]})
    files = {im["id"]: im["file_name"] for im in instances["images"]}
    urls = {im["id"]: im.get("coco_url") for im in instances["images"]}
    Path(coco_images).mkdir(parents=True, exist_ok=True)
    amber = {sha256(p) for p in Path(amber_images).iterdir() if p.is_file()}
    rng = random.Random(seed)
    rng.shuffle(with_objects)
    rows = []
    for imid in with_objects:
        path = Path(coco_images) / files[imid]
        if not path.exists() and download and urls.get(imid):
            import urllib.request
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(urls[imid], tmp)
            tmp.rename(path)
        if not path.exists() or sha256(path) in amber:
            continue
        rows.append({"id": "coco-%d" % imid, "image": str(path.resolve()), "split": "dev", "source": "COCO-val2014"})
        if len(rows) == n:
            break
    if len(rows) < n:
        raise ValueError("Only %d eligible COCO images found" % len(rows))
    atomic_json(output, rows)
    return {"images": len(rows), "seed": seed, "excluded_amber_hashes": len(amber),
            "note": "Exact-hash exclusion only; audit near duplicates separately."}


def evaluate(answers_path, instances_path, captions_path, output):
    """answers: [{"id": "coco-<imid>", "response": caption}] (e.g. an export *.captions.json)."""
    chair = Chair()
    answers = records(answers_path)
    ids = [int(str(a["id"]).split("-")[-1]) for a in answers]
    seg, truth = coco_ground_truth(instances_path, captions_path, set(ids), chair)
    rows, n_obj, n_hall, n_hcap, cover_num, cover_den, words = [], 0, 0, 0, 0, 0, 0
    for a, imid in zip(answers, ids):
        found, nodes = chair.words(a["response"])
        hallucinated = [(w, n) for w, n in zip(found, nodes) if n not in truth[imid]]
        named = set(nodes)
        n_obj += len(nodes)
        n_hall += len(hallucinated)
        n_hcap += bool(hallucinated)
        cover_num += len(seg[imid] & named)
        cover_den += len(seg[imid])
        words += len(a["response"].split())
        rows.append({"id": a["id"], "hallucinated": hallucinated, "named_objects": sorted(named),
                     "segmentation_objects": sorted(seg[imid])})
    metrics = {"CHAIRs": n_hcap / len(answers), "CHAIRi": n_hall / n_obj if n_obj else 0.0,
               "Cover_seg": cover_num / cover_den if cover_den else None, "mean_words": words / len(answers),
               "captions": len(answers)}
    atomic_json(output, {"metrics": metrics, "per_image": rows,
                         "method": "CHAIR (Rohrbach et al. 2018) python3 port; WordNet lemmatizer for singularization"})
    return metrics
