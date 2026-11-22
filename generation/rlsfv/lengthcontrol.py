"Post-hoc length control for caption-selection runs."
from __future__ import annotations


def words(text):
    return len((text or "").split())


def mean(values):
    return sum(values) / len(values) if values else None


def rate(flags):
    return sum(1 for x in flags if x) / len(flags) if flags else None


def matched_mask(reference_words, candidate_words, tol=10):
    if len(reference_words) != len(candidate_words):
        raise ValueError("matched_mask needs paired per-image lists of equal length")
    return [abs(a - b) <= tol for a, b in zip(reference_words, candidate_words)]


def _subset(values, mask):
    return [v for v, keep in zip(values, mask) if keep]


def matched_delta(reference, candidate, tol=10):
    "Per-image paired delta on the subset with |word gap| <= tol."
    mask = matched_mask(reference["words"], candidate["words"], tol)
    n = sum(mask)
    if n == 0:
        return {"tol": tol, "n": 0, "d_CHAIRs": None, "d_Cover_seg": None}
    ref_hall = _subset(reference["has_hall"], mask)
    can_hall = _subset(candidate["has_hall"], mask)
    ref_num = sum(_subset(reference["cover_num"], mask))
    ref_den = sum(_subset(reference["cover_den"], mask))
    can_num = sum(_subset(candidate["cover_num"], mask))
    can_den = sum(_subset(candidate["cover_den"], mask))
    ref_cover = ref_num / ref_den if ref_den else None
    can_cover = can_num / can_den if can_den else None
    d_cover = (can_cover - ref_cover) if (ref_cover is not None and can_cover is not None) else None
    return {"tol": tol, "n": n,
            "d_CHAIRs": rate(can_hall) - rate(ref_hall),
            "d_Cover_seg": d_cover}


def stratified(methods, bins=((0, 70), (70, 80), (80, 90), (90, 100), (100, 110), (110, None))):
    """CHAIRs and Cover_seg per word-count bin. ``methods`` maps name -> per-image dict."""
    out = {}
    for name, values in methods.items():
        rows = []
        w = values["words"]
        for lo, hi in bins:
            idx = [i for i, x in enumerate(w) if x >= lo and (hi is None or x < hi)]
            if not idx:
                continue
            num = sum(values["cover_num"][i] for i in idx)
            den = sum(values["cover_den"][i] for i in idx)
            rows.append({"bin": [lo, hi], "n": len(idx),
                         "CHAIRs": rate([values["has_hall"][i] for i in idx]),
                         "Cover_seg": num / den if den else None})
        out[name] = rows
    return out


def raw(methods):
    return {name: {"n": len(v["words"]), "CHAIRs": rate(v["has_hall"]),
                   "Cover_seg": (sum(v["cover_num"]) / sum(v["cover_den"])) if sum(v["cover_den"]) else None,
                   "mean_words": mean(v["words"])}
            for name, v in methods.items()}


def summary(methods, reference, tols=(5, 10, 15), bins=((0, 70), (70, 80), (80, 90), (90, 100), (100, 110), (110, None))):
    """Return raw rates, matched deltas at several tolerances, and the stratified table."""
    if reference not in methods:
        raise KeyError("reference %r not in methods" % reference)
    ref = methods[reference]
    out = {"reference": reference, "raw": raw(methods),
           "matched": {name: [matched_delta(ref, v, tol) for tol in tols]
                       for name, v in methods.items() if name != reference},
           "stratified": stratified(methods, bins),
           "note": ("Raw and length-controlled numbers answer different questions: raw = what "
                    "deployment would show; matched = comparison among captions of similar length. "
                    "Length can be a mediator, so report both.")}
    return out


def from_rows(captions, per_image, order=None):
    "Build the per-image dict from one method's captions.json and eval-*.json."
    caps = {cap["id"]: cap for cap in captions}
    rows = {row["id"]: row for row in per_image}
    ids = list(order) if order is not None else [cap["id"] for cap in captions]
    missing = [i for i in ids if i not in caps or i not in rows]
    if missing:
        raise KeyError("missing caption/eval rows for ids: %s" % missing[:5])
    if order is not None and len(ids) != len(caps):
        raise ValueError("order has %d ids but captions has %d" % (len(ids), len(caps)))
    values = {"ids": ids, "words": [], "has_hall": [], "cover_num": [], "cover_den": []}
    for image_id in ids:
        cap, row = caps[image_id], rows[image_id]
        named = set(row.get("named_objects") or [])
        seg = set(row.get("segmentation_objects") or [])
        values["words"].append(words(cap.get("response")))
        values["has_hall"].append(1 if row.get("hallucinated") else 0)
        values["cover_num"].append(len(seg & named))
        values["cover_den"].append(len(seg))
    return values
