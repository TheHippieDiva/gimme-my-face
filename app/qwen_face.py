"""A scene with the singer's face: Qwen Image Edit 2509 (+ Lightning 8-step), up to 3 reference photos.

python3 app/qwen_face.py OUTDIR "scene" FACE1.jpg [FACE2.jpg] [FACE3.jpg]
Face photos are the user's own and are uploaded only with their OK.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

KEEP = ("Use the person from the reference photos and keep their exact face: same face shape, cheekbones, chin, eyes, "
        "nose, skin tone, glasses (if any) and smile; never older, never a different person. ")
NO_TEXT = " No text, no letters, no words, no signs, no logos, no badges."


def workflow(face_names, scene, seed=None, w=768, h=1344, keep=True, who=""):
    """who = the singer's own description of their look (hair, ink, build), added after KEEP."""
    enc = {"clip": ["2", 0], "vae": ["3", 0]}
    for i, n in enumerate(face_names[:3], 1):
        enc[f"image{i}"] = [f"img{i}", 0]
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
        "1b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "1c": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1b", 0], "shift": 3.0}},
        "1d": {"class_type": "CFGNorm", "inputs": {"model": ["1c", 0], "strength": 1.0}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {**enc, "prompt": ((KEEP + who + " ") if keep else "") + scene + NO_TEXT}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {**enc, "prompt": ""}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["1d", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0],
            "seed": seed if seed is not None else random.randint(0, 2**31), "steps": 8, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "qwen_face"}},
    }
    for i, n in enumerate(face_names[:3], 1):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": n}}
    return wf


if __name__ == "__main__":
    out, scene, faces = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    names = [comfy.upload_image(f) for f in faces]
    pid = comfy.submit(workflow(names, scene))
    print("submitted", pid)
    comfy.wait(pid, timeout=900)
    n0 = len([f for f in os.listdir(out) if f.startswith("qwen_face_")])
    for i, f in enumerate(comfy.outputs(pid)):
        print("saved", comfy.download(f, os.path.join(out, f"qwen_face_{n0 + i}.png")))
