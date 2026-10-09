"""Spoken-language detection restricted to English, Hindi and Kannada.

Samples three 30-second windows (15%, 50%, 80% of the file) so music or silence at
the start cannot fool it, runs Whisper's language head on each, and averages the
probabilities of the three allowed languages only. Hindi vs Urdu and Kannada vs
Telugu confusions disappear because those labels are not candidates.
"""
import os

CANDIDATES = ("en", "hi", "kn")
NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}


def _load_model(size):
    from faster_whisper import WhisperModel
    dev = os.environ.get("SAAR_DETECT_DEVICE", "cuda")
    if dev == "cuda":
        try:
            return WhisperModel(size, device="cuda", compute_type="int8_float16")
        except Exception as ex:
            print(f"[lang] GPU load failed ({ex}); detecting on CPU")
    return WhisperModel(size, device="cpu", compute_type="int8")


def _release(m):
    """Free the detector's GPU memory so the main transcriber can load its own model."""
    import gc
    del m
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def detect_language(path, size=None, model=None, audio=None):
    """-> {'code','name','confidence','probs','uncertain'}"""
    import numpy as np
    if audio is None:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(path, sampling_rate=16000)
    own = model is None
    m = model if model is not None else _load_model(size or os.environ.get("SAAR_DETECT_MODEL", "large-v3"))
    sr, win = 16000, 30 * 16000
    n = len(audio)
    if n <= win * 1.2:
        windows = [audio]
    else:
        windows = []
        for frac in (0.15, 0.50, 0.80):
            a = int(max(0, min(n - win, n * frac - win / 2)))
            windows.append(audio[a:a + win])
    acc = {c: 0.0 for c in CANDIDATES}
    try:
        for w in windows:
            w = np.asarray(w, dtype="float32")
            _, info = m.transcribe(w, language=None, beam_size=1, without_timestamps=True)
            probs = dict(getattr(info, "all_language_probs", None) or [(info.language, info.language_probability)])
            tot = sum(probs.get(c, 0.0) for c in CANDIDATES)
            for c in CANDIDATES:
                acc[c] += (probs.get(c, 0.0) / tot) if tot > 0 else 1.0 / len(CANDIDATES)
    finally:
        if own:
            _release(m)
        del m
    acc = {c: v / len(windows) for c, v in acc.items()}
    code = max(acc, key=acc.get)
    return {"code": code, "name": NAMES[code], "confidence": round(acc[code], 3),
            "probs": {c: round(v, 3) for c, v in acc.items()}, "uncertain": acc[code] < 0.6}