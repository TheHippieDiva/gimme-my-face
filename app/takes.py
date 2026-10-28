"""IS THIS YOU?  Three portrait takes from your face photos; you pick the one that's you, and it becomes your LOOK:
the face every scene is locked to (scenes.py --look).

    python3 app/takes.py OUT_DIR FACE1.jpg [FACE2 FACE3]          -> OUT_DIR/takes/take1..3.png
    python3 app/takes.py OUT_DIR --pick 2                          -> OUT_DIR/look.png (face crop of take 2)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402
import qwen_face  # noqa: E402

PORTRAIT = ("A natural portrait photo of this person, chest up, three-quarter angle, looking slightly off camera with a "
            "soft smile, warm soft light, plain dark background. Exactly one person.")


def make_takes(out, faces, who="", n=3, progress=print):
    d = os.path.join(out, "takes")
    os.makedirs(d, exist_ok=True)
    names = [comfy.upload_image(f) for f in faces]
    pids = [comfy.submit(qwen_face.workflow(names, PORTRAIT, seed=random.randint(0, 2**31), who=who)) for _ in range(n)]
    got = []
    for i, pid in enumerate(pids, 1):
        comfy.wait(pid, timeout=900)
        p = comfy.download(comfy.outputs(pid)[0], os.path.join(d, f"take{i}.png"))
        got.append(p)
        progress(f"take {i} of {n} ready")
    return got


def pick(out, i):
    """Face crop of the chosen take = the look (a crop, so the face pass copies a face, never a whole picture)."""
    from PIL import Image
    im = Image.open(os.path.join(out, "takes", f"take{i}.png")).convert("RGB")
    w, h = im.size
    look = os.path.join(out, "look.png")
    im.crop((int(w * 0.08), int(h * 0.06), int(w * 0.92), int(h * 0.62))).save(look)
    return look


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--pick" in a:
        print(pick(a[0], int(a[a.index("--pick") + 1])))
    else:
        print(make_takes(a[0], a[1:]))
