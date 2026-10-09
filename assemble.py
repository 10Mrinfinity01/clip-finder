"""Smooth-join video/audio assembly for Saar (replaces assemble_short in main.py for the server)."""
def assemble_short(src, edl, out_path, is_video, pad=0.1, pad_in=0.08, pad_out=0.30, xf=0.25):
    """Join KEEP runs with a soft cross-fade (audio and video overlap) instead of
    dipping to black, and leave a little room after the last word of each run."""
    keeps = [x for x in edl if x["action"] == "keep"]
    if not keeps:
        return False
    if is_video:
        from moviepy import VideoFileClip, concatenate_videoclips, vfx, afx
        with VideoFileClip(src) as clip:
            parts = []
            for k in keeps:
                s = max(0, k["start"] - pad_in)
                e = min(clip.duration, k["end"] + pad_out)
                if e - s > 2 * xf:
                    parts.append(clip.subclipped(s, e))
            if not parts:
                return False
            out = []
            for i, p in enumerate(parts):
                fx = []
                if i > 0:
                    fx += [vfx.CrossFadeIn(xf), afx.AudioFadeIn(xf)]
                else:
                    fx += [vfx.FadeIn(0.3), afx.AudioFadeIn(0.2)]
                if i < len(parts) - 1:
                    fx += [afx.AudioFadeOut(xf)]
                else:
                    fx += [vfx.FadeOut(0.4), afx.AudioFadeOut(0.4)]
                out.append(p.with_effects(fx))
            final = concatenate_videoclips(out, method="compose", padding=-xf)
            final.write_videofile(out_path, codec="libx264", audio_codec="aac", preset="veryfast", threads=4, logger=None)
    else:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src)
        out = None
        ms = int(xf * 1000)
        for k in keeps:
            seg = audio[int(max(0, k["start"] - pad_in) * 1000):int((k["end"] + pad_out) * 1000)]
            if len(seg) < 2 * ms:
                continue
            out = seg if out is None else out.append(seg, crossfade=ms)
        if out is None:
            return False
        out.fade_in(200).fade_out(400).export(out_path, format="mp3")
    return True