"""Explicit input list, with dataset-neutral filenames and a legacy-name alias."""
import json
from pathlib import Path
from .protocol import FEATURE_FILES

def image_key(value):
    """Canonical string ID; generation also accepts integer IDs, including zero."""
    if type(value) not in (str,int) or not str(value).strip():
        raise ValueError('Image ID must be a nonempty string or integer')
    return str(value)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def feature_paths(directory):
    directory = Path(directory)
    candidates = [directory/n for n in ('candidate_features.json', 'coco_features.json') if (directory/n).is_file()]
    if len(candidates) != 1:
        raise ValueError('Provide exactly one candidate_features.json (legacy coco_features.json also accepted)')
    return candidates + [directory/n for n in FEATURE_FILES]

def read_bundle(directory):
    paths = feature_paths(directory)
    feats, allf, cons, cons2 = [read_json(p) for p in paths]
    return feats, allf, {(r['image_id'],r['cand']):r for r in cons}, {(r['image_id'],r['cand']):r for r in cons2}, paths
