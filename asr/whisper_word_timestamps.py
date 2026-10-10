import os
import sys
import json
import ctypes
from pathlib import Path

CUDA_DLL_DIR = Path(
    r"C:/Users/RAKSHITH G M/Downloads/cuBLAS.and.cuDNN_CUDA12_win_v3"
)

if CUDA_DLL_DIR.exists():
    os.add_dll_directory(str(CUDA_DLL_DIR))
    ctypes.WinDLL(str(CUDA_DLL_DIR / "cublas64_12.dll"))

from faster_whisper import WhisperModel


def transcribe_with_words(audio_path, language="kn"):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print("[+] Loading Faster-Whisper large-v3...")

    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16"
    )

    print("[+] Transcribing...")
    print("[+] Word timestamps enabled")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=False
    )

    segments = list(segments)

    output_segments = []
    all_words = []

    for seg in segments:

        words = []

        if seg.words:

            for w in seg.words:

                item = {
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "word": w.word
                }

                words.append(item)
                all_words.append(item)

        output_segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "words": words
        })

    output = {
        "audio": str(audio_path),
        "language": language,
        "model": "Faster-Whisper large-v3",
        "segments": output_segments,
        "words": all_words
    }

    output_path = audio_path.with_name(
        audio_path.stem + "_whisper_words.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("================================")
    print("WHISPER WORD TIMESTAMPS COMPLETE")
    print("================================")
    print("Segments:", len(output_segments))
    print("Words:", len(all_words))
    print("Detected language:", info.language)
    print("Saved:", output_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python whisper_word_timestamps.py "
            "<audio> [language]"
        )
        sys.exit(1)

    audio = sys.argv[1]

    language = sys.argv[2] if len(sys.argv) >= 3 else "kn"

    transcribe_with_words(audio, language)
