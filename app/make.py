"""ONE BUTTON: song + lyrics + face photos -> a vertical music video with your face and karaoke captions.

    python3 app/make.py SONG.mp3 LYRICS.txt OUT_DIR FACE1.jpg [FACE2.jpg FACE3.jpg] [--vibe "..."] [--story "..."]
                        [--look LOOK.png] [--motion off|key|all] [--pin 0.7]

Steps (each one skips itself if its output already exists, so a re-run only redoes what changed):
  1. lyricbeat  : words timed to the song + the beat (on your computer, free)
  2. storyboard : Claude on Comfy turns each line into a picture idea (no lyrics ever go into a picture)
  3. scenes     : a still with your face for every slot (Qwen Image Edit on Comfy), OCR text check;
                  motion with the face pin (LTX-2 on Comfy) for key slots or all of them
  3b. faceswap  : your REAL face on every scene from your first face photo (FaceFusion, on your computer, free);
                  --realface off skips it
  4. assemble   : cuts on the bar, captions + end card as the top layer, vertical 1080x1920
GPU runs on YOUR Comfy account (your API key). `estimate()` says roughly what a run costs before you press Make.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# rough Comfy Cloud costs (Standard plan GPU ~ $4.54/h, billed per second; measured on our test runs)
COST = {"still": 0.03, "clip": 0.20, "storyboard": 0.05}


def plan_motion(slots, mode):
    """Which slots get motion: off = none, all = every slot, key = every third slot + the first and last."""
    if mode == "off":
        return set()
    if mode == "all":
        return {s["id"] for s in slots}
    return {s["id"] for i, s in enumerate(slots) if i % 3 == 0 or i == len(slots) - 1}


def estimate(n_slots, motion="key"):
    clips = {"off": 0, "all": n_slots, "key": n_slots // 3 + 1}[motion]
    return round(COST["storyboard"] + n_slots * 2 * COST["still"] + clips * COST["clip"], 2)


def log(msg, cb=None):
    print(msg, flush=True)
    if cb:
        cb(msg)


def make(song, lyrics, out, faces, vibe="dark, moody, cinematic night", story="", look=None, motion="key",
         pin=0.7, progress=None, end_card=None, who="", realface=True):
    import assemble
    import lyricbeat
    import scenes
    import storyboard
    os.makedirs(out, exist_ok=True)
    slots_path = os.path.join(out, "slots.json")
    if not os.path.exists(os.path.join(out, "timeline.json")):
        log("1/4 timing your lyrics to the song + finding the beat…", progress)
        lyricbeat.main(song, lyrics, out)
    if not os.path.exists(slots_path):
        log("2/4 writing the storyboard (Claude on Comfy)…", progress)
        spec = storyboard.main(out, slots_path, vibe, story)
        spec["who"] = who
        if end_card:
            spec["end_card"] = {"text": end_card, "secs": 4}
        json.dump(spec, open(slots_path, "w"), indent=1)
    spec = json.load(open(slots_path))
    todo = [s["id"] for s in spec["slots"] if not s.get("media")]
    if todo:
        moving = plan_motion(spec["slots"], motion)
        log(f"3/4 making {len(todo)} scenes with your face (≈ ${estimate(len(todo), motion)} of Comfy GPU)…", progress)
        still_ids = [i for i in todo if i not in moving]
        move_ids = [i for i in todo if i in moving]
        if still_ids:
            scenes.main(slots_path, out, faces, still_ids, False, pin, look)
        if move_ids:
            scenes.main(slots_path, out, faces, move_ids, True, pin, look)
    if realface and faces:
        import faceswap
        if faceswap.available():
            log("3b/4 putting your real face on every scene (FaceFusion, on your computer)…", progress)
            faceswap.main(slots_path, faces[0], progress)
        else:
            log("3b/4 skipped: FaceFusion isn't installed (see app/faceswap.py)", progress)
    log("4/4 cutting the video on the beat + captions on top…", progress)
    final = assemble.main(song, out, slots_path)
    log(f"done: {final}", progress)
    return final


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = {}
    for k in ("--vibe", "--story", "--look", "--motion", "--pin", "--end", "--realface"):
        if k in a:
            i = a.index(k)
            opt[k[2:]] = a[i + 1]
            del a[i:i + 2]
    make(a[0], a[1], a[2], a[3:], opt.get("vibe", "dark, moody, cinematic night"), opt.get("story", ""),
         opt.get("look"), opt.get("motion", "key"), float(opt.get("pin", 0.7)), end_card=opt.get("end"),
         realface=opt.get("realface", "on") != "off")
