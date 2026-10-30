"""Auto storyboard: the timed lyrics + a vibe -> a slot list (one picture idea per line), written by Claude on Comfy.

    python3 app/storyboard.py OUT_DIR SLOTS.json "vibe in a few words" ["who you are / story notes"]

Reads OUT_DIR/timeline.json (lyricbeat.py). Lines become slots; gaps with no singing (intro, breaks, outro) get
their own slots. Claude (the ClaudeNode on Comfy Cloud, billed to the same Comfy account) turns each line into a
PICTURE: the singer in a scene, described plainly. It never quotes the lyrics, so no words end up in an image.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

MODEL = "Sonnet 5.5"
RULES = """You storyboard a vertical music video for the singer of this song. The singer's own face is added later.
The singer is ONE character for the whole video (continuity is the bible): same face, age, gender, build, hair and ONE outfit
in every scene, unless the story clearly calls for a costume change.
First line: OUTFIT | one outfit for the whole video, in plain words (fits the singer, the music and the vibe)
Then ONLY one line per item below, same order, nothing else, in this exact form:
NN | what (5-12 word note) | prompt | move
- NN = the item number (01, 02, ...). No "|" inside the fields.
- prompt: one or two plain sentences describing a single photo: the singer (call them "the person"; one person only unless
  the story needs a second person, then describe that one without a face close-up), where she is, what she does, light, mood.
  Make it fit the vibe and tell the song's story visually; keep a through-line (same place/props recurring).
- NEVER put lyrics, words, letters, signs, logos, brand names, screens with readable text, or real people's names in a prompt.
- Safe for work. No children's faces.
- move: one short sentence of SMALL motion for a 3-second clip (a glance, a smile, hair moving, light sweeping, slow push-in).
"""


def items(tl, gap=3.0):
    """Lines and the instrumental gaps between them, as (at, text-or-None)."""
    out, t = [], 0.0
    for ln in tl["lines"]:
        if ln["start"] - t >= gap:
            out.append((round(t, 2), None))
        if out and out[-1][1] and ln["start"] - out[-1][0] < 4.0:
            # a scene lasts at least ~4 s: short lines share one picture (29 Oct: 28 scenes in 16 s took 28 min)
            out[-1] = (out[-1][0], out[-1][1] + " / " + ln["text"])
        else:
            out.append((round(ln["start"] - 0.25, 2), ln["text"]))
        t = ln["end"]
    out.append((round(t + 0.5, 2), None))       # outro
    return out


def ask_claude(prompt, system):
    wf = {"1": {"class_type": "ClaudeNode", "inputs": {"prompt": prompt, "system_prompt": system, "model": MODEL,
                                                         "model.max_tokens": 8000, "model.reasoning_effort": "off", "seed": 1}},
          "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}
    pid = comfy.submit(wf)
    comfy.wait(pid, timeout=600)
    return comfy.job(pid)["outputs"]["2"]["text"][0]


def main(out, slots_path, vibe, notes="", who=""):
    tl = json.load(open(f"{out}/timeline.json"))
    its = items(tl)
    listing = "\n".join(f'{i + 1:02d}. at={a}: ' + (f'line: "{t}"' if t else "(instrumental, no singing)")
                        for i, (a, t) in enumerate(its))
    ask = (f"Vibe: {vibe}\nThe singer (the same in every scene): {who or '(not given)'}\n"
           f"The story: {notes or '(none: tell it from the lyrics)'}\n\nItems:\n{listing}")
    raw = ask_claude(ask, RULES)
    open(slots_path + ".raw.txt", "w").write(raw)
    got, outfit = {}, ""
    for ln in raw.splitlines():
        f = [x.strip() for x in ln.split("|")]
        if len(f) >= 2 and f[0].upper() == "OUTFIT":
            outfit = f[1]
            continue
        if len(f) >= 4 and f[0].isdigit():
            got[int(f[0])] = f
    spec = {"slots": []}
    for i, (a, t) in enumerate(its, 1):
        f = got.get(i) or got.get(max(k for k in got if k < i) if any(k < i for k in got) else min(got))
        spec["slots"].append({"id": f"{i:02d}", "at": a, "what": f[1], "line": t,
                              "prompt": re.sub(r'["“”].*?["“”]', "", f[2]),   # belt and braces: drop anything quoted
                              "move": f[3], "status": "placeholder", "media": []})
    spec.update(name=os.path.basename(out.rstrip("/")) + "_video", end_card=None, vibe=vibe, outfit=outfit)
    json.dump(spec, open(slots_path, "w"), indent=1)
    print(f"{len(spec['slots'])} slots -> {slots_path}")
    return spec


if __name__ == "__main__":
    main(*sys.argv[1:5])
