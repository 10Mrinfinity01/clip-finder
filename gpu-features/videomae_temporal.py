import torch
import numpy as np
import json
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch.nn.functional as F
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to input video")
args = parser.parse_args()

VIDEO = str(Path(args.video).resolve())

video_path = Path(VIDEO)
OUTPUT = str(
    video_path.parent / "gpu-features" /
    f"{video_path.stem}_videomae_scores.json"
)



device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading video...")
vr = VideoReader(VIDEO, ctx=cpu(0))

fps = float(vr.get_avg_fps())
total_frames = len(vr)
duration = total_frames / fps

print(f"FPS: {fps:.2f}")
print(f"Duration: {duration:.2f}s")

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
model = model.to(device)
model.eval()

# Analyze one window every 2 seconds.
WINDOW_SECONDS = 2.0
STEP_SECONDS = 2.0
NUM_FRAMES = 16

results = []
previous_feature = None

start_time = 0.0

while start_time + WINDOW_SECONDS <= duration:

    # 16 frames distributed across the 2-second window
    times = np.linspace(
        start_time,
        start_time + WINDOW_SECONDS,
        NUM_FRAMES,
        endpoint=False
    )

    indices = np.clip(
        (times * fps).astype(int),
        0,
        total_frames - 1
    )

    frames = vr.get_batch(indices).asnumpy()

    inputs = processor(list(frames), return_tensors="pt")
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool VideoMAE token features
    feature = outputs.last_hidden_state.mean(dim=1)

    # Normalize feature
    feature = F.normalize(feature, dim=-1)

    # Compare this window with previous window
    if previous_feature is None:
        change = 0.0
    else:
        similarity = torch.sum(
            previous_feature * feature
        ).item()

        change = 1.0 - similarity

    results.append({
        "start": round(start_time, 2),
        "end": round(start_time + WINDOW_SECONDS, 2),
        "temporal_change": round(float(change), 6)
    })

    previous_feature = feature

    print(
        f"{start_time:6.1f}s - "
        f"{start_time + WINDOW_SECONDS:6.1f}s | "
        f"change={change:.4f}"
    )

    start_time += STEP_SECONDS

# Normalize changes to 0-100
changes = np.array(
    [x["temporal_change"] for x in results],
    dtype=np.float32
)

if len(changes) > 1:
    low = float(changes.min())
    high = float(changes.max())

    if high > low:
        scores = 100 * (changes - low) / (high - low)
    else:
        scores = np.zeros_like(changes)
else:
    scores = np.zeros_like(changes)

for item, score in zip(results, scores):
    item["temporal_change_score"] = round(float(score), 2)

output = {
    "video": VIDEO,
    "duration": round(duration, 2),
    "window_seconds": WINDOW_SECONDS,
    "step_seconds": STEP_SECONDS,
    "model": "MCG-NJU/videomae-base",
    "results": results
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nDONE!")
print(f"Saved: {OUTPUT}")

print("\nTop temporal-change moments:")

top = sorted(
    results,
    key=lambda x: x["temporal_change_score"],
    reverse=True
)[:10]

for x in top:
    print(
        f"{x['start']:6.1f}s - "
        f"{x['end']:6.1f}s : "
        f"{x['temporal_change_score']}"
    )
