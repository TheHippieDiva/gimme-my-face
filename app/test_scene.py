"""First test: one still scene through Comfy Cloud (Flux schnell, no face yet).

python3 app/test_scene.py out/test
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

NO_TEXT = "no text, no letters, no words, no captions, no signs, no logos, no badges"
PROMPT = ("Cinematic still, dark garage at night, a single hanging work light flickers on; a dusty canvas tarp is "
          "sliding off a rusted classic 1960s American fastback muscle car, dust swirling in the light beam, "
          "moody blue-black shadows with warm amber glow, 35mm film look, shallow depth of field. " + NO_TEXT)


def workflow(prompt, w=720, h=1280, seed=None):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": seed if seed is not None else random.randint(0, 2**31), "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "scene01"}},
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out/test"
    os.makedirs(out, exist_ok=True)
    pid = comfy.submit(workflow(PROMPT))
    print("submitted", pid)
    comfy.wait(pid)
    for i, f in enumerate(comfy.outputs(pid)):
        print("saved", comfy.download(f, os.path.join(out, f"scene01_{i}.png")))
