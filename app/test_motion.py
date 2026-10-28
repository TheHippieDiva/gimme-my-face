"""Second test: animate a still scene with LTX-Video (image -> ~4 s clip).

python3 app/test_motion.py out/test/scene01_0.png out/test
Frames come back as PNGs and are joined into an MP4 here with ffmpeg.
"""
import glob
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

PROMPT = ("The dusty canvas tarp slowly slides off the rusted 1960s fastback muscle car, dust swirls and glitters in "
          "the beam of the single hanging work light, the light flickers once, slow cinematic camera push-in, "
          "dark garage, moody amber and blue-black. No text, no letters, no logos.")
NEG = "text, letters, words, watermark, logo, badge, blurry, distorted, jittery, low quality, cartoon"


def workflow(image_name, prompt, w=512, h=896, frames=97, fps=24, seed=None):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-video-2b-v0.9.5.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp16.safetensors", "type": "ltxv"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "LTXVPreprocess", "inputs": {"image": ["5", 0], "img_compression": 35}},
        "7": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "vae": ["1", 2], "image": ["6", 0],
            "width": w, "height": h, "length": frames, "batch_size": 1, "strength": 1.0}},
        "8": {"class_type": "LTXVConditioning", "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": fps}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "LTXVScheduler", "inputs": {"steps": 30, "max_shift": 2.05, "base_shift": 0.95,
                                                         "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
        "11": {"class_type": "SamplerCustom", "inputs": {
            "model": ["1", 0], "add_noise": True, "noise_seed": seed if seed is not None else random.randint(0, 2**31),
            "cfg": 3.0, "positive": ["8", 0], "negative": ["8", 1], "sampler": ["9", 0], "sigmas": ["10", 0],
            "latent_image": ["7", 2]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "motion"}},
    }


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    name = comfy.upload_image(src)
    print("uploaded", name)
    pid = comfy.submit(workflow(name, PROMPT))
    print("submitted", pid)
    comfy.wait(pid)
    files = sorted(comfy.outputs(pid), key=lambda f: f.get("display_name", f["filename"]))
    for i, f in enumerate(files):
        comfy.download(f, os.path.join(fdir, f"f{i:04d}.png"))
    print("frames", len(files))
    mp4 = os.path.join(out, "scene01_motion.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", os.path.join(fdir, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    print("saved", mp4)
