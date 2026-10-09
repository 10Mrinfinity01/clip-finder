import os
import re
import json
import uuid
import hashlib
import threading
import traceback

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

import main as core
import fusion
import signals_loader as sl
import emotion
import edit_modes as em
import lang_detect
import assemble

app = Flask(__name__)
CORS(app)
for d in ("uploads", "clips", "signals", "static"):
    os.makedirs(d, exist_ok=True)

JOBS = {}                      # job_id -> job dict (kept in memory)
GPU_LOCK = threading.Lock()    # one heavy job at a time (GPU / Whisper)


# ---------------------------------------------------------------- helpers
def _new_job(kind, **extra):
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "stage": "queued",
           "error": None, "result": None}
    job.update(extra)
    JOBS[job["id"]] = job
    return job


def _run(job, fn):
    def worker():
        try:
            with GPU_LOCK:
                fn(job)
            job["stage"] = "done"
        except Exception as ex:
            traceback.print_exc()
            job["stage"] = "error"
            job["error"] = str(ex)
    threading.Thread(target=worker, daemon=True).start()


def _public(job):
    return {"id": job["id"], "kind": job["kind"], "stage": job["stage"],
            "error": job["error"], "result": job["result"]}


def _duration(path, is_video):
    try:
        if is_video:
            from moviepy import VideoFileClip
            with VideoFileClip(path) as c:
                return round(c.duration, 2)
        from pydub import AudioSegment
        return round(len(AudioSegment.from_file(path)) / 1000, 2)
    except Exception:
        return None


def _signals_dirs():
    out = []
    if os.path.isdir("signals"):
        for name in sorted(os.listdir("signals")):
            p = os.path.join("signals", name)
            if os.path.isfile(os.path.join(p, "visual_fusion.json")):
                out.append(name)
    return out


# ---------------------------------------------------------------- step 1
def _analyze(job):
    path, lang, model = job["path"], job["lang"], job["model"]
    det = None
    if not lang or lang == "auto":
        job["stage"] = "detecting"
        try:
            det = lang_detect.detect_language(path)
            lang = det["code"]
            print(f"[+] Detected language: {det['name']} ({det['confidence']:.0%})")
        except Exception as ex:
            print(f"[!] Language detection failed ({ex}); letting Whisper decide.")
            lang = None

    job["stage"] = "transcribing"
    segs = core.transcribe(path, model, lang)
    if lang and lang != "en":
        job["stage"] = "translating"
        segs = core.transcribe(path, model, lang, task="translate")

    job["stage"] = "scoring"
    sents = core.build_sentences(segs)
    visual = words = None
    sd = job.get("signals_dir")
    if sd:
        visual = sl.load_visual(sd)
        words = sl.load_words(sd)
    scored = fusion.fuse(sents, visual=visual)
    top = fusion.top_moments(scored, 5)

    job["stage"] = "emotions"
    emo, esum = emotion.analyze(sents)
    by_start = {round(s["start"], 2): emo[i] for i, s in enumerate(sents)}

    def row(s):
        e = by_start.get(round(s["start"], 2)) or {}
        return {"start": round(s["start"], 2), "end": round(s["end"], 2),
                "text": s["text"], "score": s.get("score_norm", 0),
                "signals": s.get("signals", {}),
                "emotion": e.get("hit"), "emotion_p": e.get("strength", 0)}

    # keep heavy objects server-side for step 2
    job["segments"] = segs
    job["sents"] = sents
    job["emo"] = emo
    job["emo_summary"] = esum
    job["visual"] = visual
    job["words"] = words

    dur = job.get("duration") or (round(sents[-1]["end"], 2) if sents else None)
    job["result"] = {
        "job_id": job["id"],
        "video_url": f"/media/{job['id']}",
        "duration": dur,
        "language": lang or "auto",
        "language_name": lang_detect.NAMES.get(lang, lang or "auto"),
        "language_detected": det is not None,
        "language_confidence": det["confidence"] if det else None,
        "language_uncertain": bool(det and det["uncertain"]),
        "signals_used": {"visual": bool(visual), "word_snap": bool(words)},
        "emotions": esum,
        "sentences": [row(s) for s in scored],
        "top_moments": [row(s) for s in top],
        "visual_timeline": [{"start": v["start"], "end": v["end"],
                             "score": round(v["visual"], 3)} for v in (visual or [])],
    }


@app.post("/analyze")
def analyze():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify({"error": "attach a file in the 'video' field"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in core.VIDEO_EXTS and ext not in core.AUDIO_EXTS:
        return jsonify({"error": f"unsupported file type {ext}"}), 400

    lang = (request.form.get("lang") or "").strip().lower() or None
    model = request.form.get("model", "large-v3")
    signals = (request.form.get("signals") or "").strip()
    stem = os.path.splitext(secure_filename(f.filename))[0]
    if signals.lower() == "none":
        signals_dir = None
    else:
        cand = signals or stem
        p = os.path.join("signals", os.path.basename(cand))
        signals_dir = p if os.path.isfile(os.path.join(p, "visual_fusion.json")) else None

    tmp = os.path.join("uploads", f"_tmp_{uuid.uuid4().hex[:8]}{ext}")
    f.save(tmp)
    h = hashlib.sha1()
    with open(tmp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    saved = os.path.join("uploads", f"{h.hexdigest()[:16]}{ext}")
    if os.path.exists(saved):
        os.remove(tmp)
    else:
        os.replace(tmp, saved)
    is_video = ext in core.VIDEO_EXTS
    job = _new_job("analyze", path=saved, lang=lang, model=model,
                   signals_dir=signals_dir, is_video=is_video,
                   duration=_duration(saved, is_video))
    _run(job, _analyze)
    return jsonify({"job_id": job["id"], "signals_dir": signals_dir}), 202


# ---------------------------------------------------------------- step 2
NOUN = {"sadness": "sadness", "joy": "joy", "anger": "anger", "fear": "fear",
        "surprise": "surprise", "disgust": "disgust"}


def _edit(job):
    a = JOBS[job["analysis_id"]]
    use = job["use_signals"]
    visual = a.get("visual") if use else None
    words = a.get("words") if use else None
    sents, emo, esum = a["sents"], a["emo"], a["emo_summary"]
    mode, emo_name = job["mode"], job["emotion"]

    job["stage"] = "planning"
    extra = {}
    if mode == "emotion":
        plan = em.plan_emotion(sents, emo, emo_name, job["target"], visual)
        if not plan["found"]:
            have = [{"emotion": e, "sentences": esum["counts"][e]} for e in esum["present"]]
            job["result"] = {
                "edit_id": job["id"], "found": False, "mode": mode, "emotion": emo_name,
                "message": f"No {emo_name} found in this video, so no short was made.",
                "available": have}
            return
        keeps = plan["keeps"]
        extra["note"] = plan.get("note", "")
        cut_reason = lambda s, e: f"No {emo_name} content in this part."
    else:
        asked = emotion.requested_emotions(job["style"], job["requirements"])
        missing = [e for e in asked if e not in esum["present"]]
        if asked and len(missing) == len(asked):
            have = [{"emotion": e, "sentences": esum["counts"][e]} for e in esum["present"]]
            names = " or ".join(NOUN.get(e, e) for e in missing)
            job["result"] = {
                "edit_id": job["id"], "found": False, "mode": mode, "emotion": missing[0],
                "message": f"No {names} found in this video, so no short was made.",
                "available": have}
            return
        if missing:
            extra["note"] = "No " + ", ".join(NOUN.get(e, e) for e in missing) + " found in this video, so the short covers the rest."
        plan = em.plan_summary(core, sents, emo, esum, job["audience"], job["target"],
                               style=job["style"], requirements=job["requirements"], visual=visual)
        keeps = plan["keeps"]
        cut_reason = plan["cut_reason"]
        extra["story"] = plan.get("story", "")
        extra["emotions_wanted"] = plan.get("present", [])

    if not keeps:
        raise RuntimeError("The editor found nothing to keep. Try a different length or option.")
    for p in em.check_integrity(keeps, sents):
        print(f"[!] integrity: {p}")

    edl = em.finalize(sents, keeps, cut_reason)

    if words:
        job["stage"] = "snapping"
        edl = [sl.snap_edl([x], words)[0] if x["action"] == "keep" else x for x in edl]
        prev_end = 0.0                       # snapping must never make clips overlap
        for x in edl:
            if x["action"] == "keep":
                x["start"] = max(x["start"], prev_end)
                prev_end = x["end"]

    job["stage"] = "rendering"
    ext_out = ".mp4" if a["is_video"] else ".mp3"
    tag = emo_name if mode == "emotion" else job["audience"]
    out_name = f"short_{tag}_{job['id']}{ext_out}"
    ok = assemble.assemble_short(a["path"], edl, os.path.join("clips", out_name), a["is_video"])
    if not ok:
        raise RuntimeError("The editor returned no KEEP segments. Try again or change the options.")

    kept = sum(x["end"] - x["start"] for x in edl if x["action"] == "keep")
    in_short = sorted({emo[i]["hit"] for i, s in enumerate(sents) if emo[i]["hit"]
                       and any(x["action"] == "keep" and x["start"] < s["end"] and x["end"] > s["start"] for x in edl)})
    with open(os.path.join("clips", f"edl_{job['id']}.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": a["path"], "mode": mode, "emotion": emo_name,
                   "audience": job["audience"], "edl": edl}, fh, indent=2, ensure_ascii=False)
    job["result"] = {
        "edit_id": job["id"], "found": True, "mode": mode, "emotion": emo_name,
        "video_url": f"/clips/{out_name}",
        "kept_seconds": round(kept, 1),
        "target_seconds": job["target"],
        "audience": job["audience"],
        "style": job["style"],
        "requirements": job["requirements"],
        "signals_used": bool(visual),
        "emotions_in_short": in_short,
        "emotions_in_video": esum["present"],
        "edl": [{"start": round(x["start"], 2), "end": round(x["end"], 2),
                 "action": x["action"], "role": x.get("role", ""),
                 "reason": x.get("reason", "")} for x in edl],
        **extra,
    }


@app.post("/edit")
def edit():
    body = request.get_json(silent=True) or {}
    a = JOBS.get(body.get("job_id", ""))
    if not a or a["kind"] != "analyze":
        return jsonify({"error": "unknown job_id"}), 404
    if a["stage"] != "done":
        return jsonify({"error": f"analysis not finished (stage: {a['stage']})"}), 409

    mode = str(body.get("mode", "summary")).lower()
    if mode not in ("summary", "emotion"):
        return jsonify({"error": "mode must be 'summary' or 'emotion'"}), 400
    emo_name = str(body.get("emotion", "")).lower()
    if mode == "emotion" and emo_name not in emotion.EMOTIONS:
        return jsonify({"error": "emotion must be one of: " + ", ".join(emotion.EMOTIONS)}), 400

    audience = re.sub(r"[^A-Za-z0-9_-]", "", str(body.get("audience", "general")))[:24] or "general"
    try:
        target = float(body.get("target_seconds", 45))
    except (TypeError, ValueError):
        target = 45.0
    target = min(max(target, 10.0), 180.0)

    job = _new_job("edit", analysis_id=a["id"], audience=audience, target=target,
                   mode=mode, emotion=emo_name if mode == "emotion" else "",
                   style=str(body.get("style", ""))[:200],
                   requirements=str(body.get("requirements", ""))[:500],
                   use_signals=bool(body.get("use_signals", True)))
    _run(job, _edit)
    return jsonify({"edit_id": job["id"]}), 202


@app.get("/debug/emotions")
def debug_emotions():
    """Plain-text view of what the emotion model thinks of every sentence in the latest analysis."""
    jobs = [j for j in JOBS.values() if j["kind"] == "analyze" and j.get("emo")]
    if not jobs:
        return "No analysed video yet.", 404, {"Content-Type": "text/plain; charset=utf-8"}
    a = jobs[-1]
    lines = [f"backend={a['emo_summary'].get('backend')} hit>={emotion.HIT_P} ctx>={emotion.CTX_P}",
             f"found: {a['emo_summary']['counts']}", ""]
    for s, e in zip(a["sents"], a["emo"]):
        top3 = sorted(e["p"].items(), key=lambda kv: -kv[1])[:3]
        t = " ".join(f"{k}:{v:.2f}" for k, v in top3)
        m, sec = divmod(int(s["start"]), 60)
        lines.append(f"{m}:{sec:02d} {'HIT-' + e['hit'] if e['hit'] else '      -'} | {t} | {s['text'][:90]}")
    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/emotions")
def emotions_list():
    return jsonify({"emotions": emotion.EMOTIONS, "backend": emotion.backend()})


# ---------------------------------------------------------------- status + files
@app.get("/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(_public(job))


@app.get("/signals")
def list_signals():
    return jsonify({"available": _signals_dirs()})


@app.get("/media/<job_id>")
def media(job_id):
    job = JOBS.get(job_id)
    if not job or "path" not in job:
        abort(404)
    return send_from_directory("uploads", os.path.basename(job["path"]))


@app.get("/clips/<name>")
def clips(name):
    return send_from_directory("clips", name)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
