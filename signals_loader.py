import json, os

def _load(p):
    return json.load(open(p, encoding="utf-8"))

def load_visual(folder):
    d = _load(os.path.join(folder, "visual_fusion.json"))
    rows = d["results"]
    vals = [r["visual_score"] for r in rows]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    return [{"start": r["start"], "end": r["end"],
             "visual": (r["visual_score"] - lo) / span,
             "clip": r.get("clip_score"), "videomae": r.get("videomae_score")}
            for r in rows]

def load_words(folder):
    d = _load(os.path.join(folder, "words.json"))
    return [{"start": w["start"], "end": w["end"], "word": w["word"].strip()}
            for w in d["words"]]

def load_indic(folder):
    d = _load(os.path.join(folder, "transcript.json"))
    return d["segments"], d.get("full_transcript", "")

def snap_edl(edl, words, max_shift=0.6, pad=0.05):
    """Move each cut start to a word start and each end to a word end."""
    if not words:
        return edl
    starts = [w["start"] for w in words]
    ends = [w["end"] for w in words]
    out = []
    for it in edl:
        it = dict(it)
        if "start" in it and "end" in it:
            s = min(starts, key=lambda x: abs(x - it["start"]))
            e = min(ends, key=lambda x: abs(x - it["end"]))
            if abs(s - it["start"]) <= max_shift:
                it["start"] = max(0.0, s - pad)
            if abs(e - it["end"]) <= max_shift:
                it["end"] = e + pad
        out.append(it)
    return out