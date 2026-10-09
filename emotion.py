"""Sentence-level emotion detection for Saar.

Primary backend: a Hugging Face emotion classifier (default
j-hartmann/emotion-english-distilroberta-base: anger, disgust, fear, joy,
neutral, sadness, surprise). It reads the ENGLISH text, which for Kannada and
Hindi videos is the translate pass that the pipeline already produces.

Fallback backend: a small keyword lexicon, used only if the model cannot be
loaded (no internet on first run, missing transformers). The result says which
backend ran, so the UI can warn that accuracy is lower.
"""
import os
import re

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABELS = EMOTIONS + ["neutral"]

MODEL_ID = os.environ.get("SAAR_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
HIT_P = float(os.environ.get("SAAR_EMOTION_HIT", "0.40"))   # a sentence on its own
CTX_P = float(os.environ.get("SAAR_EMOTION_CTX", "0.45"))   # a sentence backed up by its neighbours
MIN_WORDS = 3                                              # ignore "Yes." / "Okay."

_LEX = {
    "joy": "happy glad joy joyful love loved wonderful great amazing awesome beautiful delighted excited "
           "celebrate celebration smile laugh laughed laughing proud thank thanks grateful blessed fun "
           "enjoy enjoyed cheers congratulations hooray win won victory perfect",
    "sadness": "sad sadly cry cried crying tears sorrow grief miss missed lonely alone lost loss death "
               "died dead gone heartbroken hurt pain painful depressed unhappy regret sorry mourn funeral "
               "hopeless empty broken",
    "anger": "angry anger furious rage hate hated mad outraged annoyed irritated shout shouted yell "
             "yelled damn stupid idiot blame betrayed fight cheated liar disgusted unfair",
    "fear": "afraid scared fear frightened terrified terror panic danger dangerous threat worried worry "
            "anxious nervous horror risk trapped run help escape nightmare",
    "surprise": "surprised surprise shocked shock suddenly unexpected unbelievable wow whoa omg incredible "
                "unexpectedly astonished amazed really seriously",
    "disgust": "disgusting disgust gross nasty revolting sick vomit filthy rotten awful horrible "
               "repulsive yuck",
}
_LEX = {k: set(v.split()) for k, v in _LEX.items()}

_pipe = None
_backend = None


def _load():
    global _pipe, _backend
    if _backend:
        return
    try:
        from transformers import pipeline
        try:
            import torch
            dev = 0 if torch.cuda.is_available() else -1
        except Exception:
            dev = -1
        _pipe = pipeline("text-classification", model=MODEL_ID, top_k=None, device=dev)
        _backend = "model"
        print(f"[emotion] using model {MODEL_ID} on {'GPU' if dev == 0 else 'CPU'}")
    except Exception as ex:
        print(f"[emotion] model unavailable ({ex}); using keyword fallback")
        _backend = "lexicon"


def backend():
    _load()
    return _backend


def _words(t):
    return re.findall(r"[A-Za-z']+", t.lower())


def _lex_probs(text):
    ws = _words(text)
    counts = {e: sum(w in _LEX[e] for w in ws) for e in EMOTIONS}
    tot = sum(counts.values())
    if tot == 0:
        return {**{e: 0.0 for e in EMOTIONS}, "neutral": 1.0}
    # keyword evidence is weak: cap confidence so one word never looks certain
    conf = min(0.85, 0.5 + 0.12 * tot)
    p = {e: conf * counts[e] / tot for e in EMOTIONS}
    p["neutral"] = 1.0 - conf
    return p


def classify(texts):
    """-> list of {label: prob} dicts (all 7 labels)."""
    _load()
    out = []
    if _backend == "model":
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            chunk = [t[:600] if t.strip() else "." for t in texts[i:i + BATCH]]
            res = _pipe(chunk, truncation=True, max_length=128)
            for r in res:
                p = {l: 0.0 for l in LABELS}
                for d in r:
                    lab = d["label"].lower()
                    if lab in p:
                        p[lab] = float(d["score"])
                out.append(p)
    else:
        out = [_lex_probs(t) for t in texts]
    return out


def analyze(sents, classify_fn=None):
    """sents: [{'start','end','text'}]. Returns (per_sentence, summary).

    per_sentence[i] = {'p': {label: prob}, 'top': label, 'hit': emotion or None,
                       'strength': prob of 'hit' (or of top)}
    A sentence is a HIT for emotion E only if E is the single strongest label
    (neutral included), E's probability >= HIT_P, and the sentence has >= MIN_WORDS words.
    """
    fn = classify_fn or classify
    probs = fn([s["text"] for s in sents]) if sents else []
    n = len(probs)
    per = []
    for i, (s, p) in enumerate(zip(sents, probs)):
        enough = len(_words(s["text"])) >= MIN_WORDS
        # context: a scene's emotion carries across neighbouring lines
        prev_p = probs[i - 1] if i > 0 else p
        next_p = probs[i + 1] if i < n - 1 else p
        ctx = {e: 0.6 * p[e] + 0.2 * prev_p[e] + 0.2 * next_p[e] for e in EMOTIONS}
        top = max(EMOTIONS, key=lambda e: p[e])           # strongest real emotion, neutral excluded
        ctop = max(EMOTIONS, key=lambda e: ctx[e])
        hit = None
        if enough and p[top] >= HIT_P and p[top] >= 0.5 * p["neutral"]:
            hit = top
        elif enough and ctx[ctop] >= CTX_P and p[ctop] >= 0.25:
            hit = ctop
        label = max(p, key=p.get)
        per.append({"p": {k: round(v, 3) for k, v in p.items()}, "top": label, "hit": hit,
                    "strength": round(max(p[hit], ctx[hit]) if hit else p[label], 3)})
    counts = {e: 0 for e in EMOTIONS}
    seconds = {e: 0.0 for e in EMOTIONS}
    first = {}
    for s, e in zip(sents, per):
        if e["hit"]:
            counts[e["hit"]] += 1
            seconds[e["hit"]] += max(0.0, s["end"] - s["start"])
            first.setdefault(e["hit"], round(s["start"], 2))
    present = sorted([e for e in EMOTIONS if counts[e] > 0], key=lambda e: -seconds[e])
    summary = {"counts": counts, "seconds": {k: round(v, 1) for k, v in seconds.items()},
               "present": present, "first": first,
               "backend": backend() if classify_fn is None else "custom",
               "hit_threshold": HIT_P, "ctx_threshold": CTX_P}
    return per, summary


# ---- emotions the user NAMES in free text (Style / Must include) -----------------
_SYN = {
    "sadness":  "sad sadness sorrow sorrowful grief grieving cry crying tears tearful heartbroken heartbreak tragic tragedy melancholy",
    "joy":      "joy joyful happy happiness cheerful delight delighted",
    "anger":    "anger angry rage furious fury outrage outraged mad",
    "fear":     "fear scary scared afraid terrifying terrified horror frightening",
    "surprise": "surprise surprised surprising shocking shocked unexpected astonishing",
    "disgust":  "disgust disgusting disgusted gross revolting",
}
_NEG = {"no", "not", "without", "avoid", "never", "non", "skip", "exclude", "dont", "don't", "except"}


def requested_emotions(*texts):
    """Emotions the user asked for in free text, e.g. Style 'sad' -> ['sadness'].
    A word with no/not/without/avoid/never in the two words before it is ignored."""
    found = []
    for text in texts:
        toks = re.findall(r"[a-z][a-z'\-]*", (text or "").lower())
        for i, t in enumerate(toks):
            if any(n in _NEG for n in toks[max(0, i - 2):i]):
                continue
            for emo, words in _SYN.items():
                if t in words.split() and emo not in found:
                    found.append(emo)
    return found
