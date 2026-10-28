"""Cut YOU out of your own video (no green screen): BiRefNet background removal on Comfy Cloud.

    python3 app/cutout.py CLIP.mov OUT_DIR [--start S --dur D]

Writes OUT_DIR/_in.mp4 (you, trimmed) + OUT_DIR/mask.mp4 (white = you), same size and frame rate. fx.inside() puts you in a world.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402


def cutout(clip, out, start=0.0, dur=None, fps=30):
    os.makedirs(out, exist_ok=True)
    trimmed = os.path.join(out, "_in.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", clip]
    if dur:
        cmd += ["-t", str(dur)]
    subprocess.run(cmd + ["-vf", f"fps={fps},scale=720:-2", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", trimmed], check=True)
    name = comfy.upload_image(trimmed)
    wf = {   # BiRefNet (open weights, built into Comfy): one mask per frame, white = you
        "1": {"class_type": "LoadVideo", "inputs": {"file": name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "4": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["3", 0], "image": ["2", 0]}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "fps": float(fps)}},
        "7": {"class_type": "SaveVideo", "inputs": {"video": ["6", 0], "filename_prefix": "video/mask", "format": "auto", "format.codec": "auto"}},
    }
    pid = comfy.submit(wf)
    print("submitted", pid)
    comfy.wait(pid, timeout=1800)
    got = {"fg": trimmed}
    for f in comfy.outputs(pid):
        got["mask"] = comfy.download(f, os.path.join(out, "mask.mp4"))
    return got


if __name__ == "__main__":
    a = sys.argv[1:]
    s = float(a[a.index("--start") + 1]) if "--start" in a else 0.0
    d = float(a[a.index("--dur") + 1]) if "--dur" in a else None
    print(cutout(a[0], a[1], s, d))
