import json
import sys
from pathlib import Path

MAX_WORD_DURATION = 3.0

if len(sys.argv) < 2:
    print("Usage: python sanitize_whisper_words.py <whisper_words.json>")
    sys.exit(1)

input_path = Path(sys.argv[1])

with open(input_path, encoding="utf-8") as f:
    data = json.load(f)

words = data["words"]

valid = []
rejected = []

for word in words:
    start = float(word["start"])
    end = float(word["end"])
    duration = end - start

    if duration <= 0:
        rejected.append({
            **word,
            "reason": "invalid_duration"
        })
    elif duration > MAX_WORD_DURATION:
        rejected.append({
            **word,
            "reason": "word_duration_too_long"
        })
    else:
        valid.append(word)

output = {
    "source": str(input_path),
    "model": data.get("model", "Faster-Whisper large-v3"),
    "language": data.get("language"),
    "max_word_duration": MAX_WORD_DURATION,
    "total_words": len(words),
    "valid_words": len(valid),
    "rejected_words": len(rejected),
    "words": valid,
    "rejected": rejected
}

output_path = input_path.with_name(
    input_path.stem + "_sanitized.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print("================================")
print("WHISPER TIMESTAMP SANITIZATION")
print("================================")
print("Total words:", len(words))
print("Valid words:", len(valid))
print("Rejected words:", len(rejected))
print("Threshold:", MAX_WORD_DURATION, "seconds")
print("Saved:", output_path)

print("\nRejected timestamps:")

for word in rejected:
    print(
        f"{word['start']:.2f}-{word['end']:.2f} "
        f"({word['end']-word['start']:.2f}s) "
        f"{word['word']}"
    )
