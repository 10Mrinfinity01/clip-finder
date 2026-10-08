import os
import sys
import json
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}

def transcribe(file_path, model_size="base"):
    print(f"[+] Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"[+] Transcribing {file_path} ...")
    segments, info = model.transcribe(file_path, word_timestamps=True)
    segments = list(segments)
    print(f"[+] Detected language: {info.language} (confidence {info.language_probability:.2f})")
    return segments

def find_matches(segments, query):
    query = query.lower().strip()
    matches = []
    for seg in segments:
        if not seg.words:
            continue
        for word in seg.words:
            cleaned = word.word.lower().strip(" .,!?\"'")
            if query == cleaned or query in cleaned:
                matches.append((word.start, word.end, word.word.strip(), "Matched by literal keyword search (Agnes unavailable)."))
    return matches

def ask_agnes(segments, query, model="agnes-3.0-flash"):
    transcript_lines = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                transcript_lines.append(f"[{w.start:.2f}-{w.end:.2f}] {w.word.strip()}")
        else:
            transcript_lines.append(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}")
    transcript_text = "\n".join(transcript_lines)

    system_prompt = (
        "You are a video editing assistant for a clip-finder tool. Given a word-level "
        "timestamped transcript and a search intent, identify the best matching moment(s). "
        "Preserve speaker intent and never select a cut that misrepresents meaning. "
        "Respond ONLY with valid JSON: a list of objects with keys 'start' (float seconds), "
        "'end' (float seconds), 'text' (the relevant quote), and 'reason' (why this was chosen, "
        "one sentence). No prose outside the JSON."
    )

    user_prompt = f"Search intent: \"{query}\"\n\nTranscript:\n{transcript_text}"

    resp = requests.post(
        f"{AGNES_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AGNES_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    print(f"[DEBUG] Raw Agnes content: {content!r}")

    decisions = json.loads(content)

    # Handle the model wrapping the list in an object, e.g. {"matches": [...]}
    if isinstance(decisions, dict):
        for key in ("matches", "decisions", "edits", "results"):
            if key in decisions and isinstance(decisions[key], list):
                decisions = decisions[key]
                break
        else:
            decisions = [decisions]

        matches = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        matches.append((float(d["start"]), float(d["end"]), d.get("text", ""), d.get("reason", "")))
    return matches

def cut_video_clip(src, start, end, out_path):
    from moviepy import VideoFileClip
    with VideoFileClip(src) as clip:
        sub = clip.subclipped(start, end)
        sub.write_videofile(out_path, codec="libx264", audio_codec="aac")

def cut_audio_clip(src, start, end, out_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(src)
    sub = audio[start * 1000:end * 1000]
    sub.export(out_path, format="mp3")

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <media_file> <search_phrase> [model_size] [pad_before] [pad_after]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"
    pad_before = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    pad_after = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in VIDEO_EXTS
    is_audio = ext in AUDIO_EXTS

    if not (is_video or is_audio):
        print(f"[!] Unsupported file type: {ext}")
        sys.exit(1)

    segments = transcribe(file_path, model_size)

    print("\n--- Full transcript ---")
    for seg in segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")
    print("------------------------\n")

    try:
        matches = ask_agnes(segments, query)
        print(f"[+] Agnes found {len(matches)} match(es).")
    except Exception as e:
        print(f"[!] Agnes API failed ({e}), falling back to keyword match.")
        matches = find_matches(segments, query)

    if not matches:
        print(f"[!] No matches found for '{query}'.")
        return

    os.makedirs("clips", exist_ok=True)
    print(f"[+] Found {len(matches)} match(es). Extracting clips...")

    edit_log = []

    for i, (start, end, text, reason) in enumerate(matches):
        padded_start = max(0, start - pad_before)
        padded_end = end + pad_after
        ext_out = ".mp4" if is_video else ".mp3"
        out_filename = f"clip_{i+1}{ext_out}"
        out_path = os.path.join("clips", out_filename)

        print(f"  -> [{padded_start:.2f}s - {padded_end:.2f}s] \"{text}\"")
        print(f"     Why: {reason}")

        if is_video:
            cut_video_clip(file_path, padded_start, padded_end, out_path)
        else:
            cut_audio_clip(file_path, padded_start, padded_end, out_path)

        edit_log.append({
            "clip": out_filename,
            "query": query,
            "source_file": file_path,
            "start": round(padded_start, 2),
            "end": round(padded_end, 2),
            "text": text,
            "reason": reason,
        })

    log_path = os.path.join("clips", "edit_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(edit_log, f, indent=2)

    print(f"[+] Done. Clips saved in ./clips/")
    print(f"[+] Edit decision log saved to {log_path}")
if __name__ == "__main__":
    main()