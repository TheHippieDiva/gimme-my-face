"""REAL FACE: put your real face on every finished scene with FaceFusion (runs on your computer, free).

Comfy makes the scenes; generated faces drift, so this last pass swaps your face from ONE clear front photo onto
every still and clip. Recipe from Dawn's tests (21 Oct 2026): one front photo only (a profile photo smears the
eyelids), hyperswap_1a_256, pixel boost 512, box + occlusion masks, NO face enhancer (it ages and fills the face).
A swap keeps the nose of the face underneath, so fix the nose in the picture first if it matters.

    python3 app/faceswap.py out/mysong/slots.json FACE.jpg

Needs FaceFusion installed (https://github.com/facefusion/facefusion); set FACEFUSION_DIR if it isn't ~/tools/facefusion.
Licensing: the hyperswap/inswapper models are non-commercial. Fine for personal videos; a paid product needs a license.
"""
import json
import os
import subprocess
import sys

FF = os.path.expanduser(os.environ.get("FACEFUSION_DIR", "~/tools/facefusion"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available():
    return os.path.exists(os.path.join(FF, "facefusion.py")) and os.path.exists(os.path.join(FF, ".venv", "bin", "python"))


def swap(target, face, out):
    """One still or clip -> its swapped copy. Returns out, or None if no face was found."""
    cmd = [os.path.join(FF, ".venv", "bin", "python"), "facefusion.py", "headless-run",
           "--source-paths", face, "--target-path", target, "--output-path", out,
           "--processors", "face_swapper", "--face-swapper-model", "hyperswap_1a_256",
           "--face-swapper-pixel-boost", "512x512", "--face-selector-mode", "one",
           "--face-selector-order", "large-small", "--face-mask-types", "box", "occlusion",
           "--output-video-quality", "90", "--execution-providers", "coreml"]
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", ""))
    subprocess.run(cmd, cwd=FF, env=env, capture_output=True, text=True)
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else None


def main(slots_path, face, progress=None):
    """Swap every slot's media. Originals stay in `media_orig`; the swapped copies go to <out>/swap/."""
    spec = json.load(open(slots_path))
    work = os.path.join(os.path.dirname(slots_path), "swap")
    os.makedirs(work, exist_ok=True)
    done = 0
    for sl in spec["slots"]:
        orig = sl.get("media_orig") or sl.get("media") or []
        new = []
        for m in orig:
            src = m if os.path.isabs(m) else os.path.join(ROOT, m)
            name, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(work, f"{sl['id']}_{name}_SWAP{ext}")
            if not os.path.exists(dst):
                swap(src, face, dst)
                done += 1
                if progress:
                    progress(f"real face on scene {sl['id']}")
            new.append(dst if os.path.exists(dst) else src)   # no face found (a hands shot): keep the original
        if orig:
            sl["media_orig"], sl["media"] = orig, new
    json.dump(spec, open(slots_path, "w"), indent=1)
    return done


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2]), "swapped")
