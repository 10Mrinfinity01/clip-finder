import os, uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import main as core

app = Flask(__name__)
CORS(app)
os.makedirs("uploads", exist_ok=True)
os.makedirs("clips", exist_ok=True)

@app.post("/process")
def process():
    f = request.files["video"]
    lang = request.form.get("lang") or None
    model = request.form.get("model", "large-v3")
    audience = request.form.get("audience", "general")
    target = float(request.form.get("target_seconds", 45))

    path = os.path.join("uploads", f"{uuid.uuid4().hex}_{f.filename}")
    f.save(path)

    segs = core.transcribe(path, model, lang)
    if lang and lang != "en":
        segs = core.transcribe(path, model, lang, "translate")

    edl = core.ask_agnes_edl(segs, audience=audience, target_seconds=target)
    out_name = f"short_{audience}_{uuid.uuid4().hex[:6]}.mp4"
    core.assemble_short(path, edl, os.path.join("clips", out_name), True)

    return jsonify({
        "video_url": f"/clips/{out_name}",
        "edl": edl,
        "transcript": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs],
    })

@app.get("/clips/<name>")
def clips(name):
    return send_from_directory("clips", name)

if __name__ == "__main__":
    app.run(port=5000, debug=False)