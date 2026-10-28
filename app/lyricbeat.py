"""Lyrics + beat: your song + your lyrics in, a timed timeline out.

    python3 app/lyricbeat.py SONG.mp3 LYRICS.txt OUT_DIR

Writes OUT_DIR/timeline.json (lines, words, beats, tempo) and OUT_DIR/captions.ass
(karaoke captions, one line at a time, words light up as they are sung).

How it works:
  * words: faster-whisper listens to the song and gives each word it hears a time. Singers
    bend words, so we line those up against YOUR lyrics (the lyrics are the truth, whisper is
    only the clock). Lyric words whisper missed get a time between their neighbours.
  * beats: a spectral-flux onset curve, the tempo from its autocorrelation, then a dynamic-
    programming beat tracker (Ellis 2007). numpy only, no extra installs.
"""
import json, os, re, subprocess, sys
from difflib import SequenceMatcher

import numpy as np

SR = 22050
HOP = 512


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


# ---------- lyrics ----------
def read_lyrics(path):
    """Lines of the lyric sheet, skipping [Section] tags and blank lines."""
    lines = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln or re.fullmatch(r"\[.*\]", ln):
            continue
        lines.append(ln)
    return lines


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def hear_words(song):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(song, word_timestamps=True, vad_filter=False)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]


def align(lines, heard):
    """Give every lyric word a start/end by matching it to the words whisper heard."""
    lyr = [(li, w) for li, ln in enumerate(lines) for w in ln.split()]
    a = [norm(w) for _, w in lyr]
    b = [norm(w) for w, _, _ in heard]
    times = [None] * len(lyr)
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = (heard[blk.b + k][1], heard[blk.b + k][2])
    # fill the gaps: spread unmatched words evenly between the matched neighbours
    known = [i for i, t in enumerate(times) if t]
    if not known:
        raise SystemExit("could not hear any lyric words in the song")
    for i in range(len(times)):
        if times[i]:
            continue
        prev = max([k for k in known if k < i], default=None)
        nxt = min([k for k in known if k > i], default=None)
        t0 = times[prev][1] if prev is not None else max(0.0, times[nxt][0] - 0.4 * (nxt - i + 1))
        t1 = times[nxt][0] if nxt is not None else t0 + 0.4 * (i - prev + 1)
        lo = prev if prev is not None else i - 1
        hi = nxt if nxt is not None else i + 1
        step = (t1 - t0) / (hi - lo)
        times[i] = (t0 + step * (i - lo - 1), t0 + step * (i - lo))
    # unheard words at the edge of a line hug their own line, not the gap (else a scat break or a long
    # instrumental gets stretched into one word): pack them ~0.4 s each next to the nearest heard word
    kset = set(known)
    for li in range(len(lines)):
        idx = [i for i, (l2, _) in enumerate(lyr) if l2 == li]
        heard_i = [i for i in idx if i in kset]
        if not heard_i:
            continue
        for k, i in enumerate(reversed([i for i in idx if i < heard_i[0]]), 1):
            t = times[heard_i[0]][0] - 0.4 * k
            if t > times[i][0]:
                times[i] = (t, t + 0.4)
        for k, i in enumerate([i for i in idx if i > heard_i[-1]]):
            t = times[heard_i[-1]][1] + 0.45 * k
            if t < times[i][0]:
                times[i] = (t, t + 0.45)
    out = []
    for li, ln in enumerate(lines):
        ws = [{"w": w, "start": round(times[i][0], 2), "end": round(times[i][1], 2)}
              for i, (l2, w) in enumerate(lyr) if l2 == li]
        out.append({"text": ln, "start": ws[0]["start"], "end": ws[-1]["end"], "words": ws,
                    "heard": sum(1 for i, (l2, _) in enumerate(lyr) if l2 == li and i in known)})
    return out


# ---------- beats ----------
def onset_curve(y):
    n = 2048
    win = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (n // 2, n // 2)), n)[::HOP] * win
    mag = np.log1p(10 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")   # remove the slow trend
    return np.maximum(flux, 0) / (flux.std() + 1e-9)


def tempo_of(env, lo=70, hi=180):
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    ok = (bpm >= lo) & (bpm <= hi)
    # prefer tempos near 110 bpm a little (log-gaussian), like most beat trackers
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 110) / 1.0) ** 2)
    score = np.where(ok, ac * weight, 0)
    lag = int(np.argmax(score))
    return 60 * fps / lag, lag


def track_beats(env, period, tightness=100):
    """Ellis' dynamic-programming beat tracker."""
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    for t in range(n):
        lo, hi = t - int(2 * period), t - int(period / 2)
        if hi <= 0:
            continue
        prev = np.arange(max(lo, 0), hi)
        pen = -tightness * np.log((t - prev) / period) ** 2
        cand = score[prev] + pen
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] += cand[k]
            back[t] = prev[k]
    # start from the best frame in the last period and walk back
    t = int(np.argmax(score[-int(period):])) + n - int(period)
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1])


def beats_of(song):
    y = decode(song)
    env = onset_curve(y)
    bpm, lag = tempo_of(env)
    frames = track_beats(env, lag)
    return round(bpm, 2), [round(f * HOP / SR, 3) for f in frames]


# ---------- captions ----------
def ass_time(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def write_ass(lines, path, w=1080, h=1920):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,78,&H0036C2FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,2,2,70,70,330,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for i, ln in enumerate(lines):
        start = ln["start"] - 0.25
        nxt = lines[i + 1]["start"] - 0.05 if i + 1 < len(lines) else ln["end"] + 1.5
        end = min(max(ln["end"] + 0.6, start + 1.0), max(nxt, start + 0.5))
        parts, cur = [], start
        for wd in ln["words"]:
            gap = max(0, wd["start"] - cur)
            if gap > 0.02:
                parts.append(f"{{\\k{int(gap * 100)}}}")
            parts.append(f"{{\\kf{max(1, int((wd['end'] - wd['start']) * 100))}}}{wd['w']} ")
            cur = wd["end"]
        ev.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Karaoke,,0,0,0,,{''.join(parts).strip()}")
    open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")


def main(song, lyrics, out):
    os.makedirs(out, exist_ok=True)
    lines = read_lyrics(lyrics)
    cache = f"{out}/heard.json"                     # the slow part: listen once, re-align as often as you like
    if os.path.exists(cache) and json.load(open(cache))["song"] == os.path.basename(song):
        heard = [tuple(h) for h in json.load(open(cache))["words"]]
    else:
        heard = hear_words(song)
        json.dump({"song": os.path.basename(song), "words": heard}, open(cache, "w"))
    timed = align(lines, heard)
    bpm, beats = beats_of(song)
    json.dump({"song": os.path.basename(song), "bpm": bpm, "beats": beats, "lines": timed},
              open(f"{out}/timeline.json", "w"), indent=1)
    write_ass(timed, f"{out}/captions.ass")
    heard_n = sum(l["heard"] for l in timed); total = sum(len(l["words"]) for l in timed)
    print(f"{len(timed)} lines, {heard_n}/{total} lyric words heard, {bpm} bpm, {len(beats)} beats -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
