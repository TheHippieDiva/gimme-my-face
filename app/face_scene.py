"""A scene with the singer's face: Flux Kontext edit from one face photo (identity kept from the reference).

python3 app/face_scene.py FACE.jpg OUTDIR "what the scene is" [seed]
Face photos are the user's own and are uploaded only with their OK.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

KEEP = ("Keep this exact woman's face and identity: same eyes, glasses, nose, smile and face shape, olive skin. "
        "She has long dreadlocks tied up in a top knot. ")
NO_TEXT = " No text, no letters, no words, no signs, no logos, no badges."


def workflow(face_name, scene, seed=None, steps=24):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev-kontext_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": face_name}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": KEEP + scene + NO_TEXT, "clip": ["2", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
        "9": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["8", 0], "guidance": 2.5}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["11", 0],
            "seed": seed if seed is not None else random.randint(0, 2**31), "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "face_scene"}},
    }


if __name__ == "__main__":
    face, out, scene = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    pid = comfy.submit(workflow(comfy.upload_image(face), scene, seed))
    print("submitted", pid)
    comfy.wait(pid, timeout=900)
    for i, f in enumerate(comfy.outputs(pid)):
        print("saved", comfy.download(f, os.path.join(out, f"face_scene_{i}.png")))
