"""Edit ONLY the tattoos on an existing scene (keeps face, hair, pose, light). Qwen Image Edit 2509, base image as latent.

python3 app/fix_ink.py SCENE.png INK_REF.jpg OUT.png "ink description"
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402
import qwen_face  # noqa: E402

scene, ink, out, desc = sys.argv[1:5]
a, b = comfy.upload_image(scene), comfy.upload_image(ink)
wf = qwen_face.workflow([a, b], "")
prompt = ("Edit image 1. Change ONLY the tattoos on her arm to match the real tattoo style in image 2: " + desc +
          " Keep everything else in image 1 exactly the same: same face, same smile, same glasses, same hair, same pose, "
          "same clothes, same car, same light and colors. No text, no letters, no words anywhere, including in the tattoos.")
wf["6"]["inputs"]["prompt"] = prompt
wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["img1", 0], "vae": ["3", 0]}}
wf["9"]["inputs"]["latent_image"] = ["enc", 0]
wf["9"]["inputs"]["seed"] = random.randint(0, 2**31)
wf["11"]["inputs"]["filename_prefix"] = "fix_ink"
pid = comfy.submit(wf)
print("submitted", pid)
comfy.wait(pid, timeout=900)
f = comfy.outputs(pid)[0]
print("saved", comfy.download(f, out))
