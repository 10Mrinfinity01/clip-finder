import json, re

W = {"visual": 0.35, "emotion": 0.30, "audio": 0.20, "speech": 0.15}

def _overlap_mean(items, key, s, e):
    vals = [it[key] for it in items if it["end"] > s and it["start"] < e]
    return sum(vals) / len(vals) if vals else None

def _speech_score(text):
    t = text or ""
    s = 0.0
    if "?" in t: s += 0.4
    if re.search(r"\d", t): s += 0.3
    if "!" in t: s += 0.2
    s += min(len(t.split()) / 25, 0.1)
    return min(s, 1.0)
def _overlap_peak_mean(items, key, s, e):
    vals = [it[key] for it in items if it["end"] > s and it["start"] < e]
    if not vals:
        return None
    return 0.5 * max(vals) + 0.5 * sum(vals) / len(vals)

def fuse(segments, visual=None, emotions=None, audio=None):
    if audio:
        m = max(a["rms"] for a in audio) or 1
        audio = [{**a, "rms": a["rms"] / m} for a in audio]
    segments = [s for s in segments
           if (s["end"] - s["start"]) <= 15 and len((s.get("text") or "").strip()) >= 4]
    out = []
    for sg in segments:
        s, e = sg["start"], sg["end"]
        sig = {
            "visual": _overlap_peak_mean(visual or [], "visual", s, e),
            "emotion": _overlap_mean(emotions or [], "intensity", s, e),
            "audio": _overlap_mean(audio or [], "rms", s, e),
            "speech": _speech_score(sg.get("text")),
        }
        have = {k: v for k, v in sig.items() if v is not None}
        tw = sum(W[k] for k in have)
        score = sum(W[k] * v for k, v in have.items()) / tw if tw else 0
        out.append({**sg, "signals": {k: round(v, 3) for k, v in have.items()},
                    "score": round(score, 3)})
    if out:
        lo = min(o["score"] for o in out)
        hi = max(o["score"] for o in out)
        for o in out:
            o["score_norm"] = round((o["score"] - lo) / (hi - lo), 3) if hi > lo else 0.5
    return out

def top_moments(scored, n=5, min_gap=8):
    picked = []
    for sg in sorted(scored, key=lambda x: -x["score"]):
        if all(abs(sg["start"] - p["start"]) > min_gap for p in picked):
            picked.append(sg)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda x: x["start"])

def save(scored, path="clips/final_scores.json"):
    json.dump(scored, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)