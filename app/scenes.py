"""Make the scenes for a slot list: a still with YOUR face for every slot that has a `prompt`, then (optionally) motion.

    python3 app/scenes.py SLOTS.json OUT_DIR FACE1 [FACE2 FACE3] [--look LOOK.png] [--only 05,07] [--motion] [--pin 0.7]

1. Still: Qwen Image Edit 2509 builds the scene from your real face photos; then, with --look (a face crop of your
   chosen look), a second edit pass swaps in that exact face + hair and changes nothing else.
2. Text check: every still is read with on-device OCR (macOS Vision); any letters -> re-roll (up to 3 tries).
   Nothing with text in it is ever used or sent back into a model.
3. Motion (--motion): LTX-2 image -> video with the FACE PIN: the still is the first frame and a slightly tighter
   crop of the same still is guided onto the last frame, so the camera pushes in and the face has to stay yours.
The slot's `media` is updated to the new still/clip (status "ai"); re-run app/assemble.py to see it in the cut.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402
import qwen_face  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOK = ("Make a BRAND-NEW photo: a different place, pose and framing from any reference images. Exactly ONE person "
        "unless the scene says otherwise. Cinematic, vertical 9:16 frame. ")


def text_in(paths):
    """{path: [text found]} via macOS Vision (tools/textcheck.js)."""
    if not paths:
        return {}
    r = subprocess.run(["osascript", "-l", "JavaScript", os.path.join(ROOT, "tools", "textcheck.js")] +
                       [os.path.abspath(p) for p in paths], capture_output=True, text=True, check=True)
    found = {}
    for ln in r.stdout.strip().splitlines():
        p, n, txt = (ln.split("\t") + ["", ""])[:3]
        found[p] = [t for t in txt.split("|") if t] if int(n or 0) else []
    return found


FACE_FIX = ("Edit image 1 only: give the person in image 1 exactly the face, glasses, skin tone and hair of the person "
            "in image 2 (they are the same person). Keep everything else in image 1 exactly the same: pose, framing, "
            "clothes, hands, background, light.")


def copied_reference(b, a, ref):
    """True when the face pass came back looking like the reference photo instead of the scene (close-ups do this)."""
    if not ref:
        return False
    import numpy as np
    from PIL import Image
    g = lambda p: np.asarray(Image.open(p).convert("L").resize((96, 168)), float)
    B = g(b)
    return abs(B - g(ref)).mean() < abs(B - g(a)).mean()


def still(face_names, prompt, dest, tries=3, look_name=None, look_file=None, who=""):
    """A: compose the scene from the real face photos. B (look_name): edit pass that locks the face to the look."""
    for t in range(tries):
        # with a look to lock to, step A builds the scene from WORDS only (photos as refs get copied whole);
        # step B then puts the face in
        pid = comfy.submit(qwen_face.workflow([] if look_name else face_names, LOOK + prompt, who=who))
        comfy.wait(pid, timeout=900)
        comfy.download(comfy.outputs(pid)[0], dest)
        if look_name:
            os.replace(dest, dest.replace(".png", "_A.png"))
            pid = comfy.submit(qwen_face.workflow([comfy.upload_image(dest.replace(".png", "_A.png")), look_name],
                                                  FACE_FIX, keep=False))
            comfy.wait(pid, timeout=900)
            comfy.download(comfy.outputs(pid)[0], dest)
            if copied_reference(dest, dest.replace(".png", "_A.png"), look_file):
                print("  face pass copied the reference picture -> keeping the scene from step A")
                os.replace(dest.replace(".png", "_A.png"), dest)
        txt = text_in([dest])[os.path.abspath(dest)]
        if not txt:
            return dest
        print(f"  text found {txt} -> re-roll ({t + 1}/{tries})")
    os.rename(dest, dest.replace(".png", "_HAS_TEXT.png"))
    return None


def push_in_end(src, dest, zoom=0.82, cx=0.5, cy=0.42):
    """End frame for the face pin: the same still, cropped tighter around (cx, cy) = a push-in."""
    from PIL import Image
    im = Image.open(src)
    w, h = im.size
    cw, ch = int(w * zoom), int(h * zoom)
    x = min(max(int(w * cx) - cw // 2, 0), w - cw)
    y = min(max(int(h * cy) - ch // 2, 0), h - ch)
    im.crop((x, y, x + cw, y + ch)).resize((w, h), Image.LANCZOS).save(dest)
    return dest


def motion(src, out_dir, prompt, pin=0.7, focus=(0.5, 0.42)):
    import ltx2
    os.makedirs(out_dir, exist_ok=True)
    end = push_in_end(src, os.path.join(out_dir, "_end.png"), cx=focus[0], cy=focus[1])
    return ltx2.run(src, out_dir, prompt + " Slow steady camera push-in. Small natural movements only; "
                    "she stays the same person the whole time.", 512, 896, 73, pin=pin, end_still=end)


def main(slots_path, out, faces, only=None, do_motion=False, pin=0.7, look=None):
    spec = json.load(open(slots_path))
    names = [comfy.upload_image(f) for f in faces]
    look_name = comfy.upload_image(look) if look else None
    sdir = os.path.join(out, "stills")
    os.makedirs(sdir, exist_ok=True)
    for sl in spec["slots"]:
        if not sl.get("prompt") or (only and sl["id"] not in only):
            continue
        n = len([f for f in os.listdir(sdir) if f.startswith(f"{sl['id']}_")])
        dest = os.path.join(sdir, f"{sl['id']}_{n}.png")
        print(f"slot {sl['id']}: {sl['what']}")
        if sl.get("face") is False:      # no person in this one (a car, a TV, lips): words only, no face pass
            got = still([], sl["prompt"], dest)
        else:
            got = still(names, sl["prompt"], dest, look_name=look_name, look_file=look, who=spec.get("who", ""))
        if not got:
            print("  gave up (text kept appearing); slot unchanged")
            continue
        media = [os.path.relpath(got, ROOT)]
        if do_motion:
            clip = motion(got, os.path.join(out, "clips", f"{sl['id']}_{n}"), sl.get("move", sl["prompt"]), pin,
                          tuple(sl.get("focus", (0.5, 0.42))))
            media = [os.path.relpath(clip, ROOT)] + media
        sl["media"], sl["status"] = media, "ai"
        json.dump(spec, open(slots_path, "w"), indent=1)   # save after every slot: safe to stop any time
        print("  ->", media)


if __name__ == "__main__":
    a = sys.argv[1:]
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    pin = float(a[a.index("--pin") + 1]) if "--pin" in a else 0.7
    look = a[a.index("--look") + 1] if "--look" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--only", "--pin", "--look"))]
    main(pos[0], pos[1], pos[2:], only, "--motion" in a, pin, look)
