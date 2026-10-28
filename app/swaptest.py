"""Face-swap test (21 Oct 2026, Dawn: "play with face swap in comfy and see if it's got anything worthy").
Comfy Cloud has no ReActor-style swap, so these are the identity options that load there, on the same AI-me locs shot:
  faceid  : IPAdapter FaceID Plus V2 on RealVisXL, img2img over one frame of the shot (a still, cents)
  phantom : Phantom-Wan 1.3B, a new 5 s clip from her real photo + the AI-me still (subject-to-video)
FaceFusion (local, free) runs from the shell; see out/swaptest/README.md.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy

PROMPT = ("photo of a woman in her 50s with long dreadlocks and a rainbow headwrap, dark rectangular glasses, black tank top, "
          "tattooed arms, smiling softly, standing in a glowing neon orange grid room, cinematic light, sharp focus")
NEG = "cartoon, painting, blurry, deformed face, extra fingers, text, watermark, old, wrinkled, grey hair"


def faceid(frame, face, out, denoise=0.55, seed=None, weight=1.5):
    denoise, weight = float(denoise), float(weight)
    wf = {
        "ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "realvisxlV50_v50Bakedvae.safetensors"}},
        # loaded by hand: the unified FaceID loader can't find the LoRA on Comfy Cloud ("LoRA model not found")
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["ck", 0], "strength_model": 0.6,
                 "lora_name": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"}},
        "ipm": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-faceid-plusv2_sdxl.bin"}},
        "ins": {"class_type": "IPAdapterInsightFaceLoader", "inputs": {"provider": "CUDA"}},
        "cv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}},
        "face": {"class_type": "LoadImage", "inputs": {"image": comfy.upload_image(face)}},
        "ip": {"class_type": "IPAdapterFaceID", "inputs": {
            "model": ["lora", 0], "ipadapter": ["ipm", 0], "image": ["face", 0], "clip_vision": ["cv", 0], "insightface": ["ins", 0], "weight": weight, "weight_faceidv2": weight,
            "weight_type": "linear", "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        "fr": {"class_type": "LoadImage", "inputs": {"image": comfy.upload_image(frame)}},
        "sc": {"class_type": "ImageScale", "inputs": {"image": ["fr", 0], "upscale_method": "lanczos", "width": 768,
                                                     "height": 1344, "crop": "center"}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["sc", 0], "vae": ["ck", 2]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ck", 1], "text": PROMPT + ", smooth youthful skin, about 45 years old, rainbow headwrap"}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ck", 1], "text": NEG}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["ip", 0], "positive": ["pos", 0], "negative": ["neg", 0],
               "latent_image": ["enc", 0], "seed": seed if seed is not None else random.randint(0, 2**31), "steps": 28,
               "cfg": 5.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ck", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "swaptest/faceid"}},
    }
    pid = comfy.submit(wf)
    print("faceid submitted", pid, flush=True)
    comfy.wait(pid, timeout=900)
    return comfy.download(comfy.outputs(pid)[0], out)


def phantom(face, look, out, seed=None, W=480, H=832, LEN=81, steps=20):
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "Phantom-Wan-1_3B_fp16.safetensors", "weight_dtype": "default"}},
        "ms": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": 5.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "f": {"class_type": "LoadImage", "inputs": {"image": comfy.upload_image(face)}},
        "l": {"class_type": "LoadImage", "inputs": {"image": comfy.upload_image(look)}},
        "fs": {"class_type": "ImageScale", "inputs": {"image": ["f", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
        "ls": {"class_type": "ImageScale", "inputs": {"image": ["l", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
        "both": {"class_type": "ImageBatch", "inputs": {"image1": ["fs", 0], "image2": ["ls", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": PROMPT + ", she sways gently and smiles, slow camera push-in"}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEG + ", static"}},
        "ph": {"class_type": "WanPhantomSubjectToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
               "width": W, "height": H, "length": LEN, "batch_size": 1, "images": ["both", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "positive": ["ph", 0], "negative": ["ph", 2],
               "latent_image": ["ph", 3], "seed": seed if seed is not None else random.randint(0, 2**31), "steps": steps,
               "cfg": 5.0, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "cvid": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 16.0}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["cvid", 0], "filename_prefix": "video/swaptest_phantom",
                 "format": "auto", "format.codec": "auto"}},
    }
    pid = comfy.submit(wf)
    print("phantom submitted", pid, flush=True)
    comfy.wait(pid, timeout=1800)
    return comfy.download(comfy.outputs(pid)[0], out)


if __name__ == "__main__":
    which, args = sys.argv[1], sys.argv[2:]
    print({"faceid": faceid, "phantom": phantom}[which](*args))
