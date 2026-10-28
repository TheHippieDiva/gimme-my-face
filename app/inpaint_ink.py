"""Repaint ONLY a masked area (e.g. tattoos on an arm) of a finished scene; everything outside the mask is kept
pixel-for-pixel (local composite). Qwen Image Edit 2509 + latent noise mask.

python3 app/inpaint_ink.py SCENE.png MASK.png INK_REF.jpg OUT.png "what goes in the masked area"
"""
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402
import qwen_face  # noqa: E402

scene, mask, ink, out, desc = sys.argv[1:6]
a, b, mk = comfy.upload_image(scene), comfy.upload_image(ink), comfy.upload_image(mask)
wf = qwen_face.workflow([a, b], "")
wf["6"]["inputs"]["prompt"] = (desc + " Realistic healed tattoo ink in skin, matching the lighting of image 1. "
                               "No text, no letters, no words in the tattoos.")
wf["mk"] = {"class_type": "LoadImageMask", "inputs": {"image": mk, "channel": "red"}}
wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["img1", 0], "vae": ["3", 0]}}
wf["nm"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["mk", 0]}}
wf["9"]["inputs"]["latent_image"] = ["nm", 0]
wf["9"]["inputs"]["seed"] = random.randint(0, 2**31)
wf["11"]["inputs"]["filename_prefix"] = "inpaint"
pid = comfy.submit(wf)
print("submitted", pid)
comfy.wait(pid, timeout=900)
raw = out.replace(".png", "_raw.png")
comfy.download(comfy.outputs(pid)[0], raw)
base, new = Image.open(scene).convert("RGB"), Image.open(raw).convert("RGB").resize(Image.open(scene).size)
Image.composite(new, base, Image.open(mask).convert("L")).save(out)
print("saved", out)
