"""Edit planning for Saar: 'emotion' mode and 'summary' (teaser) mode.

Every run is made of WHOLE sentences (indexes first..last), so a cut never lands
mid-sentence. Both planners return {"found": bool, "keeps": [...], ...}; the
server turns the keeps into a full keep/cut EDL with finalize().
"""
import json
import re

import requests

import fusion

LO, HI = 0.85, 1.15          # accepted share of the target length
MIN_RUN = 3.0                # shortest useful clip, seconds
GAP_MERGE = 2.5              # merge clips closer than this
_CONJ = ("and ", "but ", "so ", "because ", "then ", "or ", "which ", "that ")


# ------------------------------------------------------------------ helpers
def _span(sents, a, b):
    return sents[b]["end"] - sents[a]["start"]


def _total(keeps):
    return sum(k["end"] - k["start"] for k in keeps)


def _mk(sents, a, b, role="", reason=""):
    return {"first": a, "last": b, "start": sents[a]["start"], "end": sents[b]["end"],
            "action": "keep", "role": role, "reason": reason}


def _set(sents, k, a, b):
    k["first"], k["last"] = a, b
    k["start"], k["end"] = sents[a]["start"], sents[b]["end"]


def _vis(visual, s):
    if not visual:
        return 0.0
    try:
        return float(fusion._overlap_peak_mean(visual, "visual", s["start"], s["end"]) or 0.0)
    except Exception:
        return 0.0


def _merge_adjacent(sents, keeps):
    keeps = sorted(keeps, key=lambda k: k["first"])
    out = []
    for k in keeps:
        if out and (k["first"] <= out[-1]["last"] + 1 or k["start"] - out[-1]["end"] < GAP_MERGE):
            m = out[-1]
            _set(sents, m, m["first"], max(m["last"], k["last"]))
            if k.get("reason") and k["reason"] not in m.get("reason", ""):
                m["reason"] = (m.get("reason", "") + " " + k["reason"]).strip()
        else:
            out.append(dict(k))
    return out


def covered(keeps, emo):
    got = set()
    for k in keeps:
        for i in range(k["first"], k["last"] + 1):
            if emo[i]["hit"]:
                got.add(emo[i]["hit"])
    return got


def check_integrity(keeps, sents):
    """Problems that would hurt meaning or playback. Empty list = clean."""
    bad = []
    starts = {round(s["start"], 3) for s in sents}
    ends = {round(s["end"], 3) for s in sents}
    prev = None
    for k in keeps:
        if round(k["start"], 3) not in starts or round(k["end"], 3) not in ends:
            bad.append(f"clip {k['start']:.1f}-{k['end']:.1f} does not sit on sentence edges")
        if prev and k["start"] < prev["end"]:
            bad.append(f"clips overlap or are out of order near {k['start']:.1f}s")
        if k["end"] - k["start"] < MIN_RUN - 0.01:
            bad.append(f"clip {k['start']:.1f}-{k['end']:.1f} is shorter than {MIN_RUN}s")
        prev = k
    return bad


def polish(sents, keeps, limit):
    """Meaning guards: do not open on a dangling 'and/but/so...' and do not end on an
    unfinished sentence. Extends by one sentence at most, only if the total stays under limit."""
    n = len(sents)
    for k in keeps:
        a, b = k["first"], k["last"]
        txt = sents[a]["text"].strip().lower()
        if a > 0 and txt.startswith(_CONJ) and sents[a]["start"] - sents[a - 1]["end"] < 1.5:
            if _total(keeps) + _span(sents, a - 1, a - 1) <= limit:
                a -= 1
        end = sents[b]["text"].strip()
        if b < n - 1 and end and end[-1] not in ".?!…।\"'" and sents[b + 1]["start"] - sents[b]["end"] < 1.5:
            if _total(keeps) + _span(sents, b + 1, b + 1) <= limit:
                b += 1
        if (a, b) != (k["first"], k["last"]):
            _set(sents, k, a, b)
    return _merge_adjacent(sents, keeps)


def finalize(sents, keeps, cut_reason):
    """keeps -> full EDL (keep + cut) for assemble_short and the UI."""
    final, cursor = [], 0.0
    for k in sorted(keeps, key=lambda x: x["start"]):
        if k["start"] - cursor > 0.5:
            final.append({"start": cursor, "end": k["start"], "action": "cut", "role": "",
                          "reason": cut_reason(cursor, k["start"])})
        final.append({x: k[x] for x in ("start", "end", "action", "role", "reason")})
        cursor = k["end"]
    end = sents[-1]["end"] if sents else cursor
    if end - cursor > 0.5:
        final.append({"start": cursor, "end": end, "action": "cut", "role": "",
                      "reason": cut_reason(cursor, end)})
    return final


# ------------------------------------------------------------------ emotion mode
def _shrink(sents, peak, limit):
    a = b = peak
    turn = 0
    while True:
        grew = False
        for side in ((0, 1) if turn == 0 else (1, 0)):
            if side == 0 and a > 0 and _span(sents, a - 1, b) <= limit:
                a -= 1; grew = True
            elif side == 1 and b < len(sents) - 1 and _span(sents, a, b + 1) <= limit:
                b += 1; grew = True
        turn ^= 1
        if not grew:
            return a, b


def plan_emotion(sents, emo, emotion, target, visual=None):
    hits = [i for i, e in enumerate(emo) if e["hit"] == emotion]
    if not hits:
        return {"found": False, "keeps": []}
    n = len(sents)
    lo_s, hi_s = target * LO, target * HI

    raw = []
    for i in hits:
        raw.append([max(0, i - 1), min(n - 1, i + 1), [i]])
    raw.sort(key=lambda r: r[0])
    runs = []
    for r in raw:
        if runs and r[0] <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], r[1]); runs[-1][2] += r[2]
        else:
            runs.append(r)

    cand = []
    for a, b, hs in runs:
        peak = max(hs, key=lambda i: emo[i]["strength"])
        if _span(sents, a, b) > hi_s:
            a, b = _shrink(sents, peak, hi_s)
            hs = [i for i in hs if a <= i <= b] or [peak]
        while _span(sents, a, b) < MIN_RUN and (a > 0 or b < n - 1):
            if b < n - 1 and _span(sents, a, b + 1) <= hi_s:
                b += 1
            elif a > 0 and _span(sents, a - 1, b) <= hi_s:
                a -= 1
            else:
                break
        score = (sum(emo[i]["strength"] for i in hs) / len(hs)
                 + 0.05 * min(len(hs), 4)
                 + 0.15 * max(_vis(visual, sents[i]) for i in range(a, b + 1)))
        cand.append({"a": a, "b": b, "hits": hs, "peak": peak, "score": score})

    chosen, tot = [], 0.0
    for c in sorted(cand, key=lambda c: -c["score"]):
        d = _span(sents, c["a"], c["b"])
        if tot + d <= hi_s:
            chosen.append(c); tot += d
    if not chosen:                       # every run too long even after shrinking
        c = max(cand, key=lambda c: c["score"])
        chosen, tot = [c], _span(sents, c["a"], c["b"])

    chosen.sort(key=lambda c: c["a"])
    changed = True
    while tot < lo_s and changed:        # add a little surrounding context, never past the target
        changed = False
        for idx, c in enumerate(chosen):
            for side in ("b", "a"):
                nxt = chosen[idx + 1]["a"] if idx + 1 < len(chosen) else n
                prv = chosen[idx - 1]["b"] if idx > 0 else -1
                if side == "b" and c["b"] + 1 < nxt and c["b"] - c["peak"] < 2 and c["b"] < n - 1:
                    add = _span(sents, c["b"] + 1, c["b"] + 1)
                    if tot + add <= hi_s:
                        c["b"] += 1; tot += add; changed = True
                if side == "a" and c["a"] - 1 > prv and c["peak"] - c["a"] < 2 and c["a"] > 0:
                    add = _span(sents, c["a"] - 1, c["a"] - 1)
                    if tot + add <= hi_s:
                        c["a"] -= 1; tot += add; changed = True
            if tot >= lo_s:
                break

    keeps = []
    for c in chosen:
        pk = sents[c["peak"]]
        pct = int(round(emo[c["peak"]]["strength"] * 100))
        text = pk["text"].strip()
        text = text if len(text) <= 90 else text[:87] + "..."
        keeps.append(_mk(sents, c["a"], c["b"], role=emotion,
                         reason=f"{emotion.capitalize()} moment ({pct}% sure): “{text}” "
                                "Whole sentences with their context, so the meaning stays intact."))
    keeps = _merge_adjacent(sents, keeps)
    keeps = polish(sents, keeps, hi_s * 1.08)
    note = ""
    if _total(keeps) < target * 0.6:
        note = (f"Only {_total(keeps):.0f} s of {emotion} content was found, so this short is "
                f"shorter than the {target:.0f} s you asked for. Padding it with other emotions "
                "would change what the video says.")
    return {"found": True, "keeps": keeps, "note": note}


# ------------------------------------------------------------------ summary mode
def _agnes_json(core, model, system_prompt, user_msg):
    resp = requests.post(
        f"{core.AGNES_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {core.AGNES_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "system", "content": system_prompt},
                                           {"role": "user", "content": user_msg}]},
        timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    m = re.search(r"[\[{]", content)
    if m:
        content = content[m.start():]
    data = json.loads(content)
    if isinstance(data, dict):
        return data.get("summary", ""), data.get("edl", [])
    return "", data


_agnes_call = _agnes_json     # tests replace this


def _cover_run(sents, emo, e, n_total):
    """Best self-contained clip for emotion e: strongest hit (prefer not in the last
    10% of the video so the teaser does not spoil the ending) plus its context."""
    idx = [i for i, x in enumerate(emo) if x["hit"] == e]
    early = [i for i in idx if sents[i]["end"] <= n_total * 0.9]
    pool = early or idx
    best = max(pool, key=lambda i: emo[i]["strength"])
    a, b = max(0, best - 1), min(len(sents) - 1, best + 1)
    while _span(sents, a, b) > 12 and (b > best or a < best):
        if b > best:
            b -= 1
        else:
            a += 1
    return best, a, b


def plan_summary(core, sents, emo, summary, audience, target, style="", requirements="",
                 visual=None, model="agnes-3.0-flash"):
    n = len(sents)
    lo_s, hi_s = target * LO, target * HI
    present = list(summary.get("present", []))[:5]
    vis = [_vis(visual, s) if visual else None for s in sents]

    def line(i, s):
        t = f"[{i}] ({s['start']:.1f}-{s['end']:.1f}s) {s['text']}"
        if emo[i]["hit"]:
            t += f"  <emo {emo[i]['hit']} {int(round(emo[i]['strength'] * 100))}>"
        if vis[i] is not None:
            t += f"  <visual {int(round(vis[i] * 100))}>"
        return t

    text = "\n".join(line(i, s) for i, s in enumerate(sents))
    extra = ""
    if present:
        extra += ("Some sentences end with <emo NAME N>, the emotion detected in that sentence "
                  f"and its confidence. This video contains: {', '.join(present)}. The short MUST "
                  "include at least one clear, self-contained moment for EACH of these emotions, "
                  "placed in the order they happen, so it shows the whole emotional range. ")
    if visual:
        extra += "<visual N> is how visually engaging the moment is; prefer higher values when equal. "
    if style:
        extra += f"Requested style: {style}. "
    if requirements:
        extra += f"Extra requirements from the user: {requirements}. "
    extra += "Ignore repeated or garbled lines. "
    system_prompt = (
        "You are an AI editor's copilot. The transcript below is split into numbered sentences. "
        f"Plan a teaser short of about {target:.0f} seconds for the audience '{audience}'. First "
        "write a one-sentence summary of the whole video. Then choose runs of consecutive sentences "
        "to KEEP and the rest to CUT. Rules: the short must make sense on its own; use 3 to 6 KEEP "
        "runs, each between 5 and 14 seconds; never start in the middle of a thought and never end "
        "mid-sentence; keep a question with its answer and a joke with its setup; keep chronological "
        "order; never change the speaker's meaning; do NOT reveal the ending or final resolution, so "
        "that anyone who watches the short wants to watch the whole video. The KEEP runs together must "
        f"total {lo_s:.0f} to {hi_s:.0f} seconds. Give each KEEP a role: hook, curiosity, escalation, "
        "or cta (the last run should leave the viewer wanting more). The transcript may be in any "
        "language. Respond ONLY with valid JSON: an object with 'summary' (string) and 'edl' (a list "
        "of objects with 'first' (sentence number), 'last' (sentence number, inclusive), 'action' "
        "('keep' or 'cut'), 'role', and 'reason' (one English sentence). No prose outside the JSON. "
        + extra)

    def build(items):
        edl = []
        for d in items:
            if not isinstance(d, dict):
                continue
            try:
                a, b = int(d["first"]), int(d["last"])
            except (KeyError, TypeError, ValueError):
                continue
            a, b = max(0, a), min(n - 1, b)
            if b < a:
                continue
            act = "cut" if str(d.get("action", "keep")).lower() == "cut" else "keep"
            edl.append({"first": a, "last": b, "start": sents[a]["start"], "end": sents[b]["end"],
                        "action": act, "role": d.get("role", ""), "reason": d.get("reason", "")})
        edl.sort(key=lambda x: x["start"])
        keeps = _merge_adjacent(sents, [x for x in edl if x["action"] == "keep"])
        keeps = [k for k in keeps if k["end"] - k["start"] >= MIN_RUN]
        return edl, keeps

    user = f"Transcript:\n{text}\n\nPlan the short."
    try:
        story, items = _agnes_call(core, model, system_prompt, user)
    except Exception as ex:               # network / API trouble: still deliver an emotion-covering plan
        print(f"[!] Agnes call failed ({ex}); building the plan from emotion coverage only.")
        story, items = "", []
    edl, keeps = build(items)
    kept = _total(keeps)
    if (not keeps) or kept > target * 1.35 or kept < target * 0.6:
        fix = (f"Your previous plan kept {kept:.0f} seconds in total, but the target is {target:.0f} "
               f"seconds: the KEEP runs must add up to {lo_s:.0f}-{hi_s:.0f} seconds while still "
               "covering every emotion listed. Add up the durations yourself. Previous plan: "
               f"{json.dumps(items)}. Respond ONLY with the corrected JSON.")
        try:
            _, items2 = _agnes_call(core, model, system_prompt, user + "\n\n" + fix)
            edl2, keeps2 = build(items2)
            if keeps2 and (not keeps or abs(_total(keeps2) - target) < abs(kept - target)):
                edl, keeps = edl2, keeps2
        except Exception as ex:
            print(f"[!] Revision failed ({ex}); keeping the first plan.")

    # ---- no single block may dominate: cap each run, then spread the budget over the story
    max_run = max(12.0, min(18.0, target * 0.3))
    dur_all = sents[-1]["end"] if sents else 0

    def cap_runs(ks):
        out = []
        for k in ks:
            if k["end"] - k["start"] <= max_run:
                out.append(k); continue
            a, b = k["first"], k["last"]
            hits = [i for i in range(a, b + 1) if emo[i]["hit"]]
            if a == 0 or not hits:                       # the opening: keep the hook, drop the rest
                wa, wb = a, a
                while wb < b and _span(sents, wa, wb + 1) <= max_run:
                    wb += 1
            else:                                        # otherwise centre on the strongest emotion
                peak = max(hits, key=lambda i: emo[i]["strength"])
                wa, wb = _shrink(sents, peak, max_run)
                wa, wb = max(wa, a), min(wb, b)
            nk = dict(k); _set(sents, nk, wa, wb); out.append(nk)
        return out

    keeps = cap_runs(keeps)

    def spread_fill(ks):
        tot = _total(ks)
        if tot >= lo_s:
            return ks
        taken = {i for k in ks for i in range(k["first"], k["last"] + 1)}
        cands = []
        for i, e in enumerate(emo):
            if i in taken or sents[i]["end"] > dur_all * 0.9:
                continue
            sc = e["strength"] if e["hit"] else (0.2 if sents[i]["text"].strip().endswith("?") else 0.0)
            if sc <= 0:
                continue
            near = min([abs(sents[i]["start"] - k["start"]) for k in ks] or [60.0])
            cands.append((sc + 0.15 * min(near / 60.0, 1.0), i))
        for _, i in sorted(cands, reverse=True):
            a, b = max(0, i - 1), min(n - 1, i + 1)
            if any(j in taken for j in range(a, b + 1)):
                continue
            d = _span(sents, a, b)
            if tot + d > hi_s:
                continue
            what = emo[i]["hit"] or "a key question"
            ks.append(_mk(sents, a, b, role="escalation",
                          reason=f"Added to spread the short across the story ({what})."))
            taken.update(range(a, b + 1)); tot += d
            if tot >= lo_s:
                break
        return _merge_adjacent(sents, ks)

    # ---- make sure every detected emotion is in the short
    if present:
        dur_all = sents[-1]["end"] if sents else 0
        for e in present:
            if e in covered(keeps, emo):
                continue
            best, a, b = _cover_run(sents, emo, e, dur_all)
            keeps.append(_mk(sents, a, b, role="escalation",
                             reason=f"Added so the short includes the {e} moment ({int(emo[best]['strength'] * 100)}% sure)."))
            keeps = _merge_adjacent(sents, keeps)

    keeps = spread_fill(keeps)

    def grow_context(ks):
        """Still short of the target: let clips run a little longer into the surrounding dialogue."""
        guard = 0
        while _total(ks) < lo_s and guard < 40:
            guard += 1
            grew = False
            for idx, k in enumerate(ks):
                nxt = ks[idx + 1]["first"] if idx + 1 < len(ks) else n
                if k["last"] + 1 < nxt and k["last"] + 1 < n and sents[k["last"] + 1]["end"] <= dur_all * 0.92:
                    add = _span(sents, k["last"] + 1, k["last"] + 1)
                    if _span(sents, k["first"], k["last"] + 1) <= max_run * 1.3 and _total(ks) + add <= hi_s:
                        _set(sents, k, k["first"], k["last"] + 1); grew = True
                if _total(ks) >= lo_s:
                    break
            if not grew:
                break
        return _merge_adjacent(sents, ks)

    keeps = grow_context(keeps)

    # ---- fit the length without losing an emotion or breaking a sentence
    def removable(k, ks):
        others = [x for x in ks if x is not k]
        return covered(others, emo) >= covered(ks, emo) & set(present)

    guard = 0
    while _total(keeps) > hi_s and guard < 50:
        guard += 1
        if len(keeps) > 3:
            pool = [k for k in keeps[1:-1] if removable(k, keeps)]
            if pool:
                drop = min(pool, key=lambda k: sum(emo[i]["strength"] for i in range(k["first"], k["last"] + 1)
                                                     if emo[i]["hit"]) / max(1, k["last"] - k["first"] + 1))
                keeps.remove(drop)
                continue
        # shorten the longest run from its end, sentence by sentence
        k = max(keeps, key=lambda x: x["end"] - x["start"])
        a, b = k["first"], k["last"]
        if b - a < 1 or _span(sents, a, b - 1) < 6:
            break

        def safe(idx):
            e = emo[idx]["hit"]
            if not e or e not in present:
                return True
            if any(emo[i]["hit"] == e for i in range(a, b + 1) if i != idx):
                return True
            return e in covered([x for x in keeps if x is not k], emo)

        if safe(b):
            _set(sents, k, a, b - 1)
        elif safe(a):
            _set(sents, k, a + 1, b)
        else:
            break

    keeps = polish(sents, keeps, hi_s * 1.08)
    cuts = [x for x in edl if x["action"] == "cut"]

    def cut_reason(s, e):
        mid = (s + e) / 2
        for c in cuts:
            if c["start"] <= mid <= c["end"]:
                return c["reason"]
        return "Not needed for the teaser."

    return {"found": True, "keeps": keeps, "story": story, "cut_reason": cut_reason,
            "present": present, "covered": sorted(covered(keeps, emo) & set(present))}
