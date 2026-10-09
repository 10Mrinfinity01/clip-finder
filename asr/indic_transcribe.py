import sys
import os
import subprocess
from pathlib import Path

MODEL_DIR = Path.home() / "clip-finder" / "models" / "indic-transcribe-core"
INFERENCE = MODEL_DIR / "inference.py"

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "kn": "Kannada",
    "en": "English",
}


def transcribe(audio_path, language="hi"):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not INFERENCE.exists():
        raise FileNotFoundError(f"ASR model not found: {INFERENCE}")

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Use one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    print(f"[+] Language: {SUPPORTED_LANGUAGES[language]}")
    print(f"[+] Audio: {audio_path}")
    print("[+] Using local Indic-Transcribe Core 1.2B...")
    print("[+] Transcribing...")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable,
            str(INFERENCE),
            str(audio_path),
            "--lang",
            language,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Indic-Transcribe failed with exit code {result.returncode}"
        )

    transcript = result.stdout.strip()

    print("\n===== TRANSCRIPT =====\n")
    print(transcript)
    print("\n===== END =====")

    return transcript


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python indic_transcribe.py <audio_file> <language>")
        print()
        print("Languages:")
        print("  hi = Hindi")
        print("  kn = Kannada")
        print("  en = English")
        sys.exit(1)

    transcribe(sys.argv[1], sys.argv[2])
