import os
import sys
from faster_whisper import WhisperModel

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
    query = query.lower()
    matches = []
    for seg in segments:
        if query in seg.text.lower():
            matches.append((seg.start, seg.end, seg.text.strip()))
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
        print("Usage: python main.py <media_file> <search_phrase> [model_size]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"

    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in VIDEO_EXTS
    is_audio = ext in AUDIO_EXTS

    if not (is_video or is_audio):
        print(f"[!] Unsupported file type: {ext}")
        sys.exit(1)

    segments = transcribe(file_path, model_size)
    matches = find_matches(segments, query)

    if not matches:
        print(f"[!] No matches found for '{query}'.")
        return

    os.makedirs("clips", exist_ok=True)
    print(f"[+] Found {len(matches)} match(es). Extracting clips...")

    for i, (start, end, text) in enumerate(matches):
        padded_start = max(0, start - 1)
        padded_end = end + 1
        ext_out = ".mp4" if is_video else ".mp3"
        out_path = os.path.join("clips", f"clip_{i+1}{ext_out}")

        print(f"  -> [{padded_start:.2f}s - {padded_end:.2f}s] \"{text}\"")
        if is_video:
            cut_video_clip(file_path, padded_start, padded_end, out_path)
        else:
            cut_audio_clip(file_path, padded_start, padded_end, out_path)

    print(f"[+] Done. Clips saved in ./clips/")

if __name__ == "__main__":
    main()