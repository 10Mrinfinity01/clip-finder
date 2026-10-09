import sys, os
import main as core
import fusion, signals_loader as sl

video, folder = sys.argv[1], sys.argv[2]
lang = sys.argv[3] if len(sys.argv) > 3 else None

def g(s, k):
    return s[k] if isinstance(s, dict) else getattr(s, k)

if lang and lang != "en":
    segs = core.transcribe(video, "large-v3", lang, "translate")
else:
    segs = core.transcribe(video, "large-v3", lang)

segs = [{"start": g(s, "start"), "end": g(s, "end"), "text": g(s, "text")} for s in segs]

visual = sl.load_visual(folder)
scored = fusion.fuse(segs, visual=visual)

for s in scored:
    print(f'{s["start"]:6.1f}-{s["end"]:6.1f}  score={s["score_norm"]:.2f}  '
          f'visual={s["signals"].get("visual")}  {s["text"][:60]}')

print("\nTOP MOMENTS")
for s in fusion.top_moments(scored, 5):
    print(f'{s["start"]:6.1f}-{s["end"]:6.1f}  score={s["score_norm"]:.2f}  {s["text"][:60]}')

os.makedirs("clips", exist_ok=True)
fusion.save(scored)
print("\nSaved clips/final_scores.json")