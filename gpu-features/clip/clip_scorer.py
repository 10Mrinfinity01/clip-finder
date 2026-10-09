import argparse
import json
import cv2
import torch
import open_clip
from pathlib import Path


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer, device


def score_video(video_path, sample_every=2.0):
    model, preprocess, tokenizer, device = load_clip()

    interesting_prompts = [
        "an exciting and visually interesting video frame",
        "a dramatic moment in a vlog",
        "an emotional moment",
        "a funny or surprising moment",
        "an important moment in a story",
        "a beautiful cinematic scene",
    ]

    boring_prompts = [
        "a boring video frame",
        "an empty or uninteresting scene",
        "a repetitive video frame",
        "a blurry or useless video frame",
    ]

    text_prompts = interesting_prompts + boring_prompts

    with torch.no_grad():
        text_tokens = tokenizer(text_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    duration = total_frames / fps

    results = []

    current_time = 0.0

    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, frame = cap.read()

        if not success:
            current_time += sample_every
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = preprocess(
            __import__("PIL").Image.fromarray(frame_rgb)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T)[0]

        interesting_score = similarity[:len(interesting_prompts)].mean()
        boring_score = similarity[len(interesting_prompts):].mean()

        score = float(
            torch.sigmoid((interesting_score - boring_score) * 10).item()
        )

        results.append({
            "timestamp": round(current_time, 2),
            "score": round(score, 4)
        })

        current_time += sample_every

    cap.release()

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "video": video_path,
        "device": device,
        "duration_seconds": round(duration, 2),
        "sample_every_seconds": sample_every,
        "top_moments": results[:20],
        "all_scores": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="CLIP visual-interest scorer"
    )

    parser.add_argument(
        "video",
        help="Path to the input video"
    )

    parser.add_argument(
        "--sample-every",
        type=float,
        default=2.0,
        help="Sample one frame every N seconds"
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    output_dir = video_path.parent / "gpu-features" / "clip"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_clip_scores.json"

    print(f"[+] Loading CLIP...")
    print(f"[+] Analyzing: {args.video}")

    result = score_video(
        args.video,
        args.sample_every
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[+] Analysis complete.")
    print(f"[+] Results saved to: {output_path}")
    print("\nTop visual moments:")

    for moment in result["top_moments"][:10]:
        print(
            f"  {moment['timestamp']:>8.2f}s"
            f"  score={moment['score']:.4f}"
        )


if __name__ == "__main__":
    main()