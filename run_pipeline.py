import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INDIC_PYTHON = (
    PROJECT_ROOT
    / "indic-asr-py310"
    / "Scripts"
    / "python.exe"
)

ASR_DIR = PROJECT_ROOT / "asr"
GPU_DIR = PROJECT_ROOT / "gpu-features"

TIMESTAMPED_SCRIPT = ASR_DIR / "timestamped_transcribe.py"
WHISPER_SCRIPT = ASR_DIR / "whisper_word_timestamps.py"
SANITIZE_SCRIPT = ASR_DIR / "sanitize_whisper_words.py"

CLIP_SCRIPT = GPU_DIR / "clip" / "clip_scorer.py"
VMAE_SCRIPT = GPU_DIR / "videomae_temporal.py"
FUSION_SCRIPT = GPU_DIR / "fuse_visual.py"


# ============================================================
# HELPER
# Run a Python script using the Indic environment
# ============================================================

def run_python_script(script, *arguments):

    command = [
        str(INDIC_PYTHON),
        str(script),
        *[str(x) for x in arguments]
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nERROR: Script failed:")
        print(script)
        sys.exit(1)

    return result


# ============================================================
# STAGE 1: AUDIO EXTRACTION
#
# Extract audio from the input video and convert it to:
# - mono
# - 16 kHz
# - PCM WAV
#
# This format is suitable for speech-recognition models.
# ============================================================

def extract_audio(video_path):

    audio_path = video_path.with_suffix(".wav")

    print("\n" + "=" * 60)
    print("STAGE 1: AUDIO EXTRACTION")
    print("=" * 60)

    print(f"Input video: {video_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print("ERROR: Audio extraction failed.")
        sys.exit(1)

    print(f"Audio saved: {audio_path}")

    return audio_path


# ============================================================
# STAGE 2: INDIC-TRANSCRIBE
#
# Generate the primary multilingual transcript.
#
# We do NOT manually pass Hindi/Kannada/English here.
# The existing timestamped transcription script can detect
# the language automatically.
# ============================================================

def run_indic_transcription(audio_path):

    print("\n" + "=" * 60)
    print("STAGE 2: INDIC-TRANSCRIBE")
    print("=" * 60)

    print("Running multilingual transcription...")

    run_python_script(
        TIMESTAMPED_SCRIPT,
        audio_path
    )

    transcript_path = audio_path.with_name(
        audio_path.stem + "_timestamped.json"
    )

    if not transcript_path.exists():
        print(f"ERROR: Transcript not found:")
        print(transcript_path)
        sys.exit(1)

    print(f"Transcript saved: {transcript_path}")

    return transcript_path


# ============================================================
# LANGUAGE DETECTION
#
# Read the language detected by Indic-Transcribe.
# This language will be passed to Faster-Whisper.
# ============================================================

def find_language(obj):

    if isinstance(obj, dict):

        # Most likely keys
        for key in [
            "language",
            "lang",
            "detected_language",
            "detected_lang"
        ]:
            value = obj.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in obj.values():
            found = find_language(value)

            if found:
                return found

    elif isinstance(obj, list):

        for value in obj:
            found = find_language(value)

            if found:
                return found

    return None


def get_detected_language(transcript_path):

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    language = find_language(data)

    if not language:
        print(
            "\nWARNING: Could not find language in transcript JSON."
        )
        print("Using automatic language detection for Whisper.")

        return None

    # Handle possible formats such as:
    # "Hindi" -> "hi"
    # "English" -> "en"
    # "Kannada" -> "kn"

    language_map = {
        "hindi": "hi",
        "english": "en",
        "kannada": "kn",
        "telugu": "te",
        "tamil": "ta",
        "malayalam": "ml",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati": "gu",
        "punjabi": "pa",
        "odia": "or",
        "assamese": "as",
        "urdu": "ur"
    }

    normalized = language.lower().strip()

    if normalized in language_map:
        language = language_map[normalized]
    else:
        language = normalized

    print(f"Detected language: {language}")

    return language


# ============================================================
# STAGE 3: FASTER-WHISPER WORD TIMESTAMPS
#
# Whisper is used mainly for precise word-level timing.
#
# Indic-Transcribe = WHAT was said
# Whisper = WHEN each word was said
# ============================================================

def run_whisper_words(audio_path, language):

    print("\n" + "=" * 60)
    print("STAGE 3: FASTER-WHISPER WORD TIMESTAMPS")
    print("=" * 60)

    if language:

        print(
            f"Running Faster-Whisper with language: {language}"
        )

        run_python_script(
            WHISPER_SCRIPT,
            audio_path,
            language
        )

    else:

        print(
            "Language unavailable; allowing Whisper to detect it."
        )

        run_python_script(
            WHISPER_SCRIPT,
            audio_path
        )

    # Expected normal output
    expected = audio_path.with_name(
        audio_path.stem + "_whisper_words.json"
    )

    if expected.exists():
        print(f"Word timestamps saved: {expected}")
        return expected

    # Fallback: locate any generated Whisper word file
    candidates = list(
        audio_path.parent.glob(
            f"{audio_path.stem}_whisper_words*.json"
        )
    )

    if not candidates:
        print("ERROR: Could not find Whisper word-timestamp file.")
        sys.exit(1)

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    whisper_file = candidates[0]

    print(f"Word timestamps found: {whisper_file}")

    return whisper_file


# ============================================================
# STAGE 4: SANITIZE WORD TIMESTAMPS
#
# Remove invalid/unsafe word timestamps before using them
# for future deterministic video-cut boundaries.
# ============================================================

def sanitize_words(whisper_file):

    print("\n" + "=" * 60)
    print("STAGE 4: SANITIZE WORD TIMESTAMPS")
    print("=" * 60)

    run_python_script(
        SANITIZE_SCRIPT,
        whisper_file
    )

    # Locate sanitized output.
    candidates = list(
        whisper_file.parent.glob(
            f"{whisper_file.name}*sanitized*.json"
        )
    )

    # Also support the normal Path.stem naming style.
    candidates += list(
        whisper_file.parent.glob(
            f"{whisper_file.stem}_sanitized.json"
        )
    )

    # Remove duplicates
    candidates = list(set(candidates))

    if not candidates:
        print("ERROR: Sanitized word file not found.")
        sys.exit(1)

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    sanitized_file = candidates[0]

    print(f"Sanitized words saved: {sanitized_file}")

    return sanitized_file


# ============================================================
# STAGE 5: CLIP
#
# CLIP estimates visual semantic relevance/interest of
# sampled video frames.
# ============================================================

def run_clip(video_path):

    print("\n" + "=" * 60)
    print("STAGE 5: CLIP VISUAL SCORING")
    print("=" * 60)

    run_python_script(
        CLIP_SCRIPT,
        video_path
    )

    clip_file = (
        GPU_DIR
        / "clip"
        / f"{video_path.stem}_clip_scores.json"
    )

    if not clip_file.exists():
        print("ERROR: CLIP output not found:")
        print(clip_file)
        sys.exit(1)

    print(f"CLIP scores saved: {clip_file}")

    return clip_file


# ============================================================
# STAGE 6: VIDEOMAE
#
# VideoMAE measures temporal visual changes between
# consecutive video windows.
# ============================================================

def run_videomae(video_path):

    print("\n" + "=" * 60)
    print("STAGE 6: VIDEOMAE TEMPORAL ANALYSIS")
    print("=" * 60)

    run_python_script(
        VMAE_SCRIPT,
        video_path
    )

    vmae_file = (
        GPU_DIR
        / f"{video_path.stem}_videomae_scores.json"
    )

    if not vmae_file.exists():
        print("ERROR: VideoMAE output not found:")
        print(vmae_file)
        sys.exit(1)

    print(f"VideoMAE scores saved: {vmae_file}")

    return vmae_file


# ============================================================
# STAGE 7: VISUAL FUSION
#
# Combine:
#
# VideoMAE = temporal visual change
# CLIP     = visual semantic relevance
#
# Current fusion:
# VideoMAE = 60%
# CLIP     = 40%
# ============================================================

def run_fusion(video_path):

    print("\n" + "=" * 60)
    print("STAGE 7: VISUAL FUSION")
    print("=" * 60)

    run_python_script(
        FUSION_SCRIPT,
        video_path
    )

    fusion_file = (
        GPU_DIR
        / f"{video_path.stem}_visual_fusion.json"
    )

    if not fusion_file.exists():
        print("ERROR: Visual fusion output not found:")
        print(fusion_file)
        sys.exit(1)

    print(f"Visual fusion saved: {fusion_file}")

    return fusion_file


# ============================================================
# STAGE 8: BUILD SIGNAL PACKAGE
#
# Collect the three important outputs into:
#
# signals/
#   video_name/
#       transcript.json
#       words.json
#       visual_fusion.json
# ============================================================

def build_signal_package(
    video_path,
    transcript_path,
    sanitized_words_path,
    fusion_path
):

    print("\n" + "=" * 60)
    print("STAGE 8: BUILD SIGNAL PACKAGE")
    print("=" * 60)

    signals_dir = (
        PROJECT_ROOT
        / "signals"
        / video_path.stem
    )

    signals_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    transcript_output = signals_dir / "transcript.json"
    words_output = signals_dir / "words.json"
    fusion_output = signals_dir / "visual_fusion.json"

    shutil.copy2(
        transcript_path,
        transcript_output
    )

    shutil.copy2(
        sanitized_words_path,
        words_output
    )

    shutil.copy2(
        fusion_path,
        fusion_output
    )

    print("\nSignal package created:")
    print(f"  {transcript_output}")
    print(f"  {words_output}")
    print(f"  {fusion_output}")

    return signals_dir


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generic multimodal video processing pipeline"
    )

    parser.add_argument(
        "video",
        help="Path to the input video"
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not video_path.exists():

        print(
            f"ERROR: Video not found:\n{video_path}"
        )

        sys.exit(1)

    if not INDIC_PYTHON.exists():

        print(
            "ERROR: Indic Python environment not found:"
        )

        print(INDIC_PYTHON)

        sys.exit(1)

    print("=" * 60)
    print("GENERIC MULTIMODAL VIDEO PIPELINE")
    print("=" * 60)

    print(f"Input video : {video_path}")
    print(f"Filename    : {video_path.name}")
    print(f"Extension   : {video_path.suffix}")

    print("\nInput video found successfully.")

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    audio_path = extract_audio(video_path)

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    transcript_path = run_indic_transcription(
        audio_path
    )

    # --------------------------------------------------------
    # DETECT LANGUAGE
    # --------------------------------------------------------

    language = get_detected_language(
        transcript_path
    )

    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    whisper_file = run_whisper_words(
        audio_path,
        language
    )

    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    sanitized_words = sanitize_words(
        whisper_file
    )

    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------

    clip_file = run_clip(
        video_path
    )

    # --------------------------------------------------------
    # STAGE 6
    # --------------------------------------------------------

    vmae_file = run_videomae(
        video_path
    )

    # --------------------------------------------------------
    # STAGE 7
    # --------------------------------------------------------

    fusion_file = run_fusion(
        video_path
    )

    # --------------------------------------------------------
    # STAGE 8
    # --------------------------------------------------------

    signals_dir = build_signal_package(
        video_path,
        transcript_path,
        sanitized_words,
        fusion_file
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

    print(f"\nVideo: {video_path.name}")

    if language:
        print(f"Language: {language}")

    print(f"\nSignals:")
    print(signals_dir)

    print("\nGenerated files:")

    print(
        f"  {signals_dir / 'transcript.json'}"
    )

    print(
        f"  {signals_dir / 'words.json'}"
    )

    print(
        f"  {signals_dir / 'visual_fusion.json'}"
    )

    print("\nReady for the Agnes reasoning/edit-decision layer.")


if __name__ == "__main__":
    main()