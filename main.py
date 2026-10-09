import os
import sys
import json
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import hashlib
import pickle
import time
load_dotenv()
AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
def _add_cuda_dll_dirs():
    import site, glob
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        for d in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            except Exception:
                pass

_add_cuda_dll_dirs()

_MODELS = {}
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE = "float16" if DEVICE == "cuda" else "int8"


def transcribe(file_path, model_size="base", language=None, task="transcribe"):
    global DEVICE, COMPUTE
    st = os.stat(file_path)
    raw = f"{os.path.abspath(file_path)}|{st.st_size}|{st.st_mtime}|{model_size}|{language}|{task}"
    key = hashlib.md5(raw.encode()).hexdigest()
    os.makedirs("cache", exist_ok=True)
    cache_path = os.path.join("cache", key + ".pkl")
    if os.path.exists(cache_path):
        print("[+] Using cached transcript")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def run(model):
        segs, info = model.transcribe(
            file_path, word_timestamps=True, language=language,
            task=task, beam_size=5, vad_filter=True,
        )
        return list(segs), info

    if model_size not in _MODELS:
        print(f"[+] Loading model '{model_size}' on {DEVICE}...")
        _MODELS[model_size] = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE)
    print(f"[+] Transcribing {file_path} ...")
    try:
        segments, info = run(_MODELS[model_size])
    except Exception as ex:
        if DEVICE != "cuda":
            raise
        print(f"[!] GPU failed ({ex}); falling back to CPU")
        DEVICE, COMPUTE = "cpu", "int8"
        _MODELS.clear()
        _MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = run(_MODELS[model_size])

    print(f"[+] Detected language: {info.language} (confidence {info.language_probability:.2f})")
    with open(cache_path, "wb") as f:
        pickle.dump(segments, f)
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

def agnes_post(messages, model="agnes-3.0-flash", retries=3, timeout=120):
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{AGNES_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {AGNES_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": messages},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as ex:
            last = ex
            print(f"[!] Agnes attempt {attempt + 1}/{retries} failed: {ex}")
            time.sleep(2)
    raise last

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
            
        "You are a video editing assistant. The transcript may be in any language "
        "(for example Kannada) and may contain transcription errors; infer meaning "
        "from context. Identify the 3 to 6 most distinct emotional moments across "
        "the whole video (for example joy, gratitude, excitement, concern, sadness, "
        "pride, urgency, humor). Use tone, word choice and context. Always return "
        "at least 2 moments if the transcript has any tonal variation. Respond ONLY "
        "with valid JSON: a list of objects with keys 'start' (float seconds), "
        "'end' (float seconds), 'text' (the quote in its original language), "
        "'emotion' (one English word), and 'reason' (one English sentence). "
        "No prose outside the JSON."
    
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

def ask_agnes_emotions(segments, model="agnes-3.0-flash"):
    transcript_lines = []
    for seg in segments:
         if seg.words:
            words = seg.words
            for i in range(0, len(words), 10):
                chunk = words[i:i+10]
                text = "".join(w.word for w in chunk).strip()
                transcript_lines.append(f"[{chunk[0].start:.2f}-{chunk[-1].end:.2f}] {text}")
         else:
            transcript_lines.append(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}")
    transcript_text = "\n".join(transcript_lines)

    system_prompt = (
        "You are a video editing assistant for a clip-finder tool. Given a word-level "
        "timestamped transcript, identify the distinct emotional moments across the "
        "whole video (e.g. joy, gratitude, excitement, sadness, frustration, surprise, "
        "nervousness). Only include clear, non-overlapping moments - don't invent emotion "
        "where there isn't one. Respond ONLY with valid JSON: a list of objects with keys "
        "'start' (float seconds), 'end' (float seconds), 'text' (the relevant quote), "
        "'emotion' (one short word/phrase), and 'reason' (why this moment shows that "
        "emotion, one sentence). No prose outside the JSON."
    )
    user_prompt = f"Transcript:\n{transcript_text}\n\nFind the emotional highlights."

    content = agnes_post([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]).strip()

    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    print(f"[DEBUG] Raw Agnes emotions content: {content!r}")

    decisions = json.loads(content)
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
        matches.append((
            float(d["start"]), float(d["end"]),
            d.get("text", ""), d.get("emotion", ""), d.get("reason", "")
        ))
    return matches
def build_sentences(segments, max_words=25, gap=0.8):
    sentences, cur = [], []

    def flush():
        if cur:
            text = "".join(w.word for w in cur).strip()
            sentences.append({"start": cur[0].start, "end": cur[-1].end, "text": text})
            cur.clear()

    prev_end = None
    for seg in segments:
        words = seg.words or []
        if not words:
            flush()
            sentences.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            prev_end = seg.end
            continue
        for w in words:
            if cur and prev_end is not None and (w.start - prev_end > gap):
                flush()
            cur.append(w)
            prev_end = w.end
            if w.word.strip().endswith((".", "?", "!", "।")) or len(cur) >= max_words:
                flush()
    flush()
    return sentences
def ask_agnes_edl(segments, audience="general", target_seconds=45, model="agnes-3.0-flash",
                  visual=None, style="", requirements=""):
    sents = build_sentences(segments)
    interest = [None] * len(sents)
    if visual:
        import fusion
        interest = [fusion._overlap_peak_mean(visual, "visual", s["start"], s["end"])
                    for s in sents]

    def _line(i, s):
        base = f"[{i}] ({s['start']:.1f}-{s['end']:.1f}s) {s['text']}"
        if interest[i] is not None:
            base += f"  <visual {int(round(interest[i] * 100))}>"
        return base

    transcript_text = "\n".join(_line(i, s) for i, s in enumerate(sents))
    extra = ""
    if visual:
        extra += ("Some sentences end with <visual N>, a 0-100 score of how visually engaging "
                  "the video is at that moment (from computer-vision analysis). Prefer sentences "
                  "with high visual scores for the hook and escalation, but never break the "
                  "story rules above for a high score. ")
    if style:
        extra += f"Requested style: {style}. "
    if requirements:
        extra += f"Extra requirements from the user: {requirements}. "
    extra += "Ignore repeated or garbled lines. "
    system_prompt = (
        "You are an AI editor's copilot. The transcript below is split into numbered "
        f"sentences. Plan a promotional short of about {target_seconds} seconds for the "
        f"audience '{audience}'. First write a one-sentence summary of the whole video. "
        "Then choose runs of consecutive sentences to KEEP and the rest to CUT. Rules: "
        "the short must make sense on its own and tell a coherent mini-story; prefer 2 "
        "to 4 KEEP runs, each at least 6 seconds long; never start in the middle of a "
        "thought; keep a question together with its answer and a joke together with its "
        "setup; keep chronological order; never change the speaker's meaning; stop before "
        "the payoff or ending so the viewer is curious. The KEEP runs together must total "
        f"{target_seconds*0.8:.0f} to {target_seconds*1.2:.0f} seconds. Give each KEEP a "
        "role in this order: hook, curiosity, escalation, cta (the last line should invite "
        "the viewer to watch the full video). The transcript may be in any language. "
        "Respond ONLY with valid JSON: an object with 'summary' (string) and 'edl' (a "
        "list of objects with 'first' (sentence number), 'last' (sentence number, "
        "inclusive), 'action' ('keep' or 'cut'), 'role', and 'reason' (one English "
        "sentence)). No prose outside the JSON."
    )
    system_prompt += " " + extra
    resp = requests.post(
        f"{AGNES_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n{transcript_text}\n\nPlan the short."},
        ]},
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    print(f"[DEBUG] Raw Agnes EDL content: {content!r}")

    data = json.loads(content)
    items = data
    if isinstance(data, dict):
        print(f"[+] Story: {data.get('summary', '')}")
        items = data.get("edl", [])
    edl = []
    for d in items:
        if not isinstance(d, dict):
            continue
        try:
            a, b = int(d["first"]), int(d["last"])
        except (KeyError, TypeError, ValueError):
            continue
        a, b = max(0, a), min(len(sents) - 1, b)
        if b < a:
            continue
        action = "cut" if str(d.get("action", "keep")).lower() == "cut" else "keep"
        edl.append({"start": sents[a]["start"], "end": sents[b]["end"], "action": action,
                    "role": d.get("role", ""), "reason": d.get("reason", "")})
    edl.sort(key=lambda x: x["start"])

    # merge keeps that are close together, drop tiny ones
    merged = []
    for k in [x for x in edl if x["action"] == "keep"]:
        if merged and k["start"] - merged[-1]["end"] < 2.5:
            merged[-1]["end"] = max(merged[-1]["end"], k["end"])
            merged[-1]["reason"] += " " + k["reason"]
        else:
            merged.append(dict(k))
    merged = [k for k in merged if k["end"] - k["start"] >= 3.0]

    kept = sum(k["end"] - k["start"] for k in merged)
    print(f"[+] Kept {kept:.1f}s (target {target_seconds:.0f}s)")

    cuts = [x for x in edl if x["action"] == "cut"]

    def cut_reason(s, e):
        mid = (s + e) / 2
        for c in cuts:
            if c["start"] <= mid <= c["end"]:
                return c["reason"]
        return "Not selected for this short."

    final, cursor = [], 0.0
    for k in merged:
        if k["start"] - cursor > 0.5:
            final.append({"start": cursor, "end": k["start"], "action": "cut", "role": "",
                          "reason": cut_reason(cursor, k["start"])})
        final.append(k)
        cursor = k["end"]
    total_end = sents[-1]["end"] if sents else cursor
    if total_end - cursor > 0.5:
        final.append({"start": cursor, "end": total_end, "action": "cut", "role": "",
                      "reason": cut_reason(cursor, total_end)})
    return final

def assemble_short(src, edl, out_path, is_video, pad=0.1):
    keeps = [x for x in edl if x["action"] == "keep"]
    if not keeps:
        return False
    if is_video:
        from moviepy import VideoFileClip, concatenate_videoclips, vfx, afx
        with VideoFileClip(src) as clip:
            parts = []
            for k in keeps:
                s = max(0, k["start"] - pad)
                e = min(clip.duration, k["end"] + pad)
                if e > s:
                    part = clip.subclipped(s, e).with_effects(
                        [vfx.FadeIn(0.2), vfx.FadeOut(0.2),
                         afx.AudioFadeIn(0.15), afx.AudioFadeOut(0.15)])
                    parts.append(part)
            if not parts:
                return False
            final = concatenate_videoclips(parts)
            final.write_videofile(out_path, codec="libx264", audio_codec="aac")
    else:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src)
        out = AudioSegment.empty()
        for k in keeps:
            seg = audio[int(max(0, k["start"] - pad) * 1000):int((k["end"] + pad) * 1000)]
            out += seg.fade_in(150).fade_out(150)
        out.export(out_path, format="mp3")
    return True
def cut_video_clip(src, start, end, out_path):
    from moviepy import VideoFileClip
    with VideoFileClip(src) as clip:
        start = max(0, start)
        end = min(end, clip.duration)
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

    lang = sys.argv[6] if len(sys.argv) > 6 else None
    segments = transcribe(file_path, model_size, lang)
    print("\n--- Full transcript ---")
    transcript_path = "transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as tf:
     for seg in segments:
        tf.write(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}\n")
    print(f"[+] Transcript saved to {transcript_path}")
    for seg in segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")
    print("------------------------\n")
    if query.lower().strip() == "short":
        audience = sys.argv[7] if len(sys.argv) > 7 else "general"
        target = float(sys.argv[8]) if len(sys.argv) > 8 else 45.0
        signals_dir = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9].lower() != "none" else None
        style = sys.argv[10] if len(sys.argv) > 10 else ""
        requirements = sys.argv[11] if len(sys.argv) > 11 else ""
        visual, words = None, None
        if signals_dir:
            import signals_loader as sl
            visual = sl.load_visual(signals_dir)
            words = sl.load_words(signals_dir)
            print(f"[+] Laptop B signals: {len(visual)} visual windows, {len(words)} words")
        agnes_segments = segments
        if lang and lang != "en":
            print("[+] Translating to English for story analysis...")
            agnes_segments = transcribe(file_path, model_size, lang, task="translate")
        edl = ask_agnes_edl(agnes_segments, audience, target,
                            visual=visual, style=style, requirements=requirements)
        if words:
            edl = [sl.snap_edl([x], words)[0] if x["action"] == "keep" else x for x in edl]
        os.makedirs("clips", exist_ok=True)
        print("\n--- Edit Decision List ---")
        for item in edl:
            print(f"{item['action'].upper():5} {item['start']:.2f}-{item['end']:.2f} "
                  f"[{item['role']}] {item['reason']}")
        ext_out = ".mp4" if is_video else ".mp3"
        tag = "_signals" if signals_dir else ""
        out_path = os.path.join("clips", f"short_{audience}{tag}{ext_out}")
        ok = assemble_short(file_path, edl, out_path, is_video)
        with open(os.path.join("clips", f"short_edl{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"source_file": file_path, "audience": audience,
                       "target_seconds": target, "style": style,
                       "requirements": requirements, "edl": edl},
                      f, indent=2, ensure_ascii=False)
        print(f"[+] Short saved to {out_path}" if ok else "[!] No KEEP segments returned.")
        return
    emotion_mode = query.lower().strip() == "emotions"

    if emotion_mode:
        # New path: scan the whole video for distinct emotional moments
        try:
         agnes_segments = segments
         if lang and lang != "en":
            print("[+] Translating to English for emotion analysis...")
            agnes_segments = transcribe(file_path, model_size, lang, task="translate")
         raw_matches = ask_agnes_emotions(agnes_segments)
        except Exception as ex:
            print(f"[!] Agnes emotions call failed: {ex}")
            return
        print(f"[+] Agnes found {len(raw_matches)} emotional moment(s).")
        # normalize to a common shape: (start, end, text, reason, emotion)
        matches = [(s, e, t, r, emo) for (s, e, t, emo, r) in raw_matches]
    else:
        # Existing path: single search intent
        try:
            raw_matches = ask_agnes(segments, query)
            print(f"[+] Agnes found {len(raw_matches)} match(es).")
        except Exception as ex:
            print(f"[!] Agnes API failed ({ex}), falling back to keyword match.")
            raw_matches = find_matches(segments, query)
        # normalize to the same shape, with emotion left blank
        matches = [(s, e, t, r, "") for (s, e, t, r) in raw_matches]

    if not matches:
        print(f"[!] No matches found for '{query}'.")
        return

    os.makedirs("clips", exist_ok=True)
    print(f"[+] Found {len(matches)} match(es). Extracting clips...")

    edit_log = []

    for i, (start, end, text, reason, emotion) in enumerate(matches):
        padded_start = max(0, start - pad_before)
        padded_end = end + pad_after
        ext_out = ".mp4" if is_video else ".mp3"
        tag = f"_{emotion}" if emotion else ""
        out_filename = f"clip_{i+1}{tag}{ext_out}"
        out_path = os.path.join("clips", out_filename)

        print(f"  -> [{padded_start:.2f}s - {padded_end:.2f}s] \"{text}\"")
        if emotion:
            print(f"     Emotion: {emotion}")
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
            "emotion": emotion,
        })

    log_path = os.path.join("clips", "edit_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(edit_log, f, indent=2)

    print(f"[+] Done. Clips saved in ./clips/")
    print(f"[+] Edit decision log saved to {log_path}")


if __name__ == "__main__":
    main()