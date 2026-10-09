import sys
import os
import json
from pathlib import Path

import torch
import soundfile as sf
import torchaudio

MODEL_DIR = Path.home() / "clip-finder" / "models" / "indic-transcribe-core"

sys.path.insert(0, str(MODEL_DIR))

from indic_transcribe import IndicTranscribe
from long_form import (
    split_on_silence,
    identify_long,
    _repetition,
    _collapse_repeats,
    CHUNK_MAX_NEW_TOKENS,
)

LOOP_REPETITION = 0.5
LOOP_MIN_WORDS = 20


def load_audio(path):
    data, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True
    )

    wav = torch.as_tensor(data).mean(dim=1)

    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000
        )

    return wav


def decode_chunk(asr, piece, lang, mode="native", depth=0):
    """
    Same anti-loop idea as the official long_form.py,
    but preserves recursive timing outside this function.
    """

    text = asr.transcribe(
        piece,
        lang=lang,
        mode=mode,
        max_new_tokens=CHUNK_MAX_NEW_TOKENS,
    )

    if isinstance(text, tuple):
        text = text[0]

    text = text.strip()
    words = text.split()

    looped = (
        len(words) >= LOOP_MIN_WORDS
        and _repetition(words) > LOOP_REPETITION
    )

    if not looped:
        return [(0, piece.numel(), text)]

    if depth >= 2 or piece.numel() < 4 * 16000:
        cleaned = " ".join(
            _collapse_repeats(words)
        )
        return [(0, piece.numel(), cleaned)]

    half = piece.numel() / 16000 / 2

    parts = split_on_silence(
        piece,
        target=half,
        hard_max=half + 2.0
    )

    if len(parts) < 2:
        middle = piece.numel() // 2
        parts = [
            (0, middle),
            (middle, piece.numel())
        ]

    results = []

    for x, y in parts:
        if y - x < 1600:
            continue

        sub_piece = piece[x:y]

        sub_results = decode_chunk(
            asr,
            sub_piece,
            lang,
            mode,
            depth + 1
        )

        for a, b, text in sub_results:
            results.append(
                (
                    x + a,
                    x + b,
                    text
                )
            )

    return results


def transcribe_timestamped(
    audio_path,
    language=None,
    mode="native"
):

    audio_path = Path(audio_path)

    print("[+] Loading audio:", audio_path)

    wav = load_audio(audio_path)

    duration = wav.numel() / 16000

    print(f"[+] Duration: {duration:.2f}s")

    print("[+] Loading Indic-Transcribe Core 1.2B...")

    asr = IndicTranscribe.from_pretrained(
        str(MODEL_DIR),
        device="cuda"
    )

    if language is None:
        print("[+] Detecting language...")
        language = identify_long(asr, wav)

    print("[+] Language:", language)
    print("[+] Creating natural-pause segments...")

    chunks = split_on_silence(wav)

    segments = []

    for index, (start_sample, end_sample) in enumerate(chunks):

        if end_sample - start_sample < 1600:
            continue

        start_time = start_sample / 16000
        end_time = end_sample / 16000

        print(
            f"[{index + 1}] "
            f"{start_time:.2f}s -> {end_time:.2f}s"
        )

        piece = wav[start_sample:end_sample]

        decoded = decode_chunk(
            asr,
            piece,
            language,
            mode
        )

        for local_start, local_end, text in decoded:

            text = text.strip()

            if not text:
                continue

            absolute_start = (
                start_sample + local_start
            ) / 16000

            absolute_end = (
                start_sample + local_end
            ) / 16000

            segments.append({
                "start": round(absolute_start, 3),
                "end": round(absolute_end, 3),
                "text": text
            })

    # Sort chronologically
    segments.sort(
        key=lambda x: x["start"]
    )

    full_text = " ".join(
        x["text"] for x in segments
    )

    base = audio_path.with_suffix("")

    json_path = Path(
        str(base) + "_timestamped.json"
    )

    txt_path = Path(
        str(base) + "_full_transcript.txt"
    )

    output = {
        "audio": str(audio_path),
        "duration": round(duration, 3),
        "language": language,
        "model": "Indic-Transcribe Core 1.2B",
        "sample_rate": 16000,
        "segments": segments,
        "full_transcript": full_text
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(full_text)

    print()
    print("================================")
    print("TRANSCRIPTION COMPLETE")
    print("================================")
    print("Segments:", len(segments))
    print("JSON:", json_path)
    print("TXT :", txt_path)

    print()
    print("FIRST SEGMENTS:")

    for segment in segments[:10]:
        print(
            f"{segment['start']:7.2f} - "
            f"{segment['end']:7.2f} | "
            f"{segment['text']}"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python timestamped_transcribe.py "
            "<audio> [language]"
        )
        sys.exit(1)

    audio = sys.argv[1]

    language = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else None
    )

    transcribe_timestamped(
        audio,
        language
    )
