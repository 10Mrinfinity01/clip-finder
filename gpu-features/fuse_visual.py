import json
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to input video")
args = parser.parse_args()

video_path = Path(args.video).resolve()

CLIP_FILE = video_path.parent / "gpu-features" / "clip" / f"{video_path.stem}_clip_scores.json"
VMAE_FILE = video_path.parent / "gpu-features" / f"{video_path.stem}_videomae_scores.json"
OUTPUT = video_path.parent / "gpu-features" / f"{video_path.stem}_visual_fusion.json"



with open(CLIP_FILE, encoding="utf-8") as f:
    clip_data = json.load(f)

with open(VMAE_FILE, encoding="utf-8") as f:
    vmae_data = json.load(f)


def find_records(obj):
    """Find lists of dictionaries containing time/score information."""
    found = []

    if isinstance(obj, dict):
        for value in obj.values():
            found.extend(find_records(value))

    elif isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            for x in obj:
                if any(k in x for k in ["time", "timestamp", "start"]):
                    found.append(x)

        for value in obj:
            if isinstance(value, (dict, list)):
                found.extend(find_records(value))

    return found


# -------------------------
# VideoMAE records
# -------------------------
vmae_records = vmae_data["results"]

vmae = []

for r in vmae_records:
    vmae.append({
        "time": float(r["start"]),
        "end": float(r["end"]),
        "vmae_score": float(r["temporal_change_score"])
    })


# -------------------------
# CLIP records
# -------------------------
raw_clip = find_records(clip_data)

clip = []

for r in raw_clip:
    time = None
    score = None

    for key in ["time", "timestamp", "start"]:
        if key in r:
            try:
                time = float(r[key])
                break
            except:
                pass

    for key in ["score", "visual_score", "clip_score"]:
        if key in r:
            try:
                score = float(r[key])
                break
            except:
                pass

    if time is not None and score is not None:
        clip.append({
            "time": time,
            "clip_raw": score
        })


# Remove duplicate CLIP records
unique = {}

for r in clip:
    unique[round(r["time"], 3)] = r

clip = list(unique.values())

if not clip:
    print("Could not find CLIP timestamp/score records.")
    print("CLIP JSON top-level keys:", list(clip_data.keys()))
    raise SystemExit(1)


# -------------------------
# Normalize CLIP scores
# -------------------------
clip_values = np.array(
    [x["clip_raw"] for x in clip],
    dtype=np.float32
)

cmin = float(clip_values.min())
cmax = float(clip_values.max())

if cmax > cmin:
    for x in clip:
        x["clip_score"] = 100.0 * (
            (x["clip_raw"] - cmin) / (cmax - cmin)
        )
else:
    for x in clip:
        x["clip_score"] = 50.0


# -------------------------
# Match CLIP to VideoMAE
# -------------------------
fused = []

for v in vmae:

    # Find nearest CLIP timestamp
    nearest = min(
        clip,
        key=lambda c: abs(c["time"] - v["time"])
    )

    clip_score = nearest["clip_score"]
    vmae_score = v["vmae_score"]

    # Initial fusion:
    # VideoMAE = temporal progression
    # CLIP = visual semantics
    visual_score = (
        0.60 * vmae_score +
        0.40 * clip_score
    )

    fused.append({
        "start": round(v["time"], 2),
        "end": round(v["end"], 2),
        "videomae_score": round(vmae_score, 2),
        "clip_score": round(clip_score, 2),
        "visual_score": round(visual_score, 2)
    })


# -------------------------
# Rank moments
# -------------------------
top = sorted(
    fused,
    key=lambda x: x["visual_score"],
    reverse=True
)

output = {
    "model": "VideoMAE + CLIP",
    "weights": {
        "videomae": 0.60,
        "clip": 0.40
    },
    "results": fused,
    "top_moments": top[:15]
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nFUSION COMPLETE!")
print("Saved:", OUTPUT)

print("\nTop visual-interest moments:")

for x in top[:10]:
    print(
        f"{x['start']:6.1f}s - {x['end']:6.1f}s | "
        f"VideoMAE={x['videomae_score']:6.2f} | "
        f"CLIP={x['clip_score']:6.2f} | "
        f"FINAL={x['visual_score']:6.2f}"
    )
