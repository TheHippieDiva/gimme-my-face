"""Collect an LTX-2 job that is still queued on Comfy Cloud after ltx2.py gave up waiting (27 Oct: the cloud queue
backed up and jobs sat 'pending' past the timeout). Waits up to 3 h, then saves frames + scene_ltx2.mp4 like ltx2.run.
    python3 app/fetch_job.py PROMPT_ID OUTDIR
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

pid, out = sys.argv[1], sys.argv[2]
comfy.wait(pid, timeout=10800)
fdir = os.path.join(out, "frames_ltx2"); os.makedirs(fdir, exist_ok=True)
files = sorted(comfy.outputs(pid), key=lambda f: f.get("display_name", f["filename"]))
for i, f in enumerate(files):
    comfy.download(f, os.path.join(fdir, f"f{i:04d}.png"))
mp4 = os.path.join(out, "scene_ltx2.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(fdir, "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
print("frames", len(files), "saved", mp4)
