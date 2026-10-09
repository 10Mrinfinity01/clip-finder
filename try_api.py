import sys, time, requests

BASE = "http://localhost:5000"
video = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else ""
signals = sys.argv[3] if len(sys.argv) > 3 else ""
audience = sys.argv[4] if len(sys.argv) > 4 else "general"
target = float(sys.argv[5]) if len(sys.argv) > 5 else 45
style = sys.argv[6] if len(sys.argv) > 6 else ""

def wait(job_id):
    last = None
    while True:
        r = requests.get(f"{BASE}/jobs/{job_id}").json()
        if r["stage"] != last:
            print("   stage:", r["stage"])
            last = r["stage"]
        if r["stage"] in ("done", "error"):
            return r
        time.sleep(1)

print("STEP 1: /analyze")
with open(video, "rb") as fh:
    r = requests.post(f"{BASE}/analyze", files={"video": fh},
                      data={"lang": lang, "signals": signals})
r.raise_for_status()
res = wait(r.json()["job_id"])
if res["stage"] == "error":
    sys.exit("analysis failed: " + str(res["error"]))
a = res["result"]
print(f"   {len(a['sentences'])} sentences, signals used: {a['signals_used']}")
print("   top moments:")
for m in a["top_moments"]:
    print(f"     {m['start']:6.1f}-{m['end']:6.1f}  {m['score']:.2f}  {m['text'][:60]}")

print("STEP 2: /edit")
r = requests.post(f"{BASE}/edit", json={
    "job_id": a["job_id"], "audience": audience, "target_seconds": target,
    "style": style, "requirements": "", "use_signals": True})
r.raise_for_status()
res = wait(r.json()["edit_id"])
if res["stage"] == "error":
    sys.exit("edit failed: " + str(res["error"]))
e = res["result"]
print(f"   kept {e['kept_seconds']}s of target {e['target_seconds']}s -> {e['video_url']}")
for x in e["edl"]:
    print(f"   {x['action'].upper():5} {x['start']:6.1f}-{x['end']:6.1f} [{x['role']}] {x['reason'][:70]}")