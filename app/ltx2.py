"""LTX-2 image -> video on Comfy Cloud (port of Comfy-Org's video_ltx2_i2v_distilled template to API format).

python3 app/ltx2.py STILL.png OUTDIR "what happens in the shot"
Two passes like the template: half-size draft, then 2x latent upscale + refine. Frames come back as PNGs and are
joined into an MP4 here (the song is laid on top later, so the model's own audio is ignored).
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

CKPT = "ltx-2-19b-distilled.safetensors"
TEXT_ENC = "gemma_3_12B_it_fp4_mixed.safetensors"
UPSCALER = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
FPS = 25
NO_TEXT = " No text, no letters, no words, no signs, no logos, no badges anywhere in the frame."


def workflow(image_name, prompt, w=704, h=1280, frames=121, seed=None, pin=0.0, ckpt=CKPT, end_name=None):
    """w,h = final size (multiples of 64); the draft pass runs at half size.
    pin > 0 = FACE PIN: the same still is also a guide on the LAST frame (at that strength), in both passes, so the
    shot has to come back to the real face instead of drifting into a stranger. Guide frames are cropped off after.
    end_name = a different end still (e.g. a tighter crop of the same face = a push-in that can't lose the face)."""
    seed = seed if seed is not None else random.randint(0, 2**31)
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "48": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "60": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TEXT_ENC, "ckpt_name": CKPT, "device": "default"}},
        "76": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt + NO_TEXT, "clip": ["60", 0]}},
        "112": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "22": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["112", 0], "frame_rate": FPS}},
        "98": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "99": {"class_type": "LTXVPreprocess", "inputs": {"image": ["98", 0], "img_compression": 33}},
        # pass 1: half size
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w // 2, "height": h // 2, "length": frames, "batch_size": 1}},
        "107": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["99", 0], "latent": ["43", 0], "strength": 1.0, "bypass": False}},
        "51": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": FPS, "batch_size": 1, "audio_vae": ["48", 0]}},
        "56": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["107", 0], "audio_latent": ["51", 0]}},
        "47": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["22", 0], "negative": ["22", 1], "cfg": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "113": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "41": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["47", 0], "sampler": ["8", 0], "sigmas": ["113", 0], "latent_image": ["56", 0]}},
        # pass 2: upscale 2x + refine
        "80": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["41", 0]}},
        "81": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["22", 0], "negative": ["22", 1], "latent": ["80", 0]}},
        "84": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["81", 2], "upscale_model": ["76", 0], "vae": ["1", 2]}},
        "108": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["99", 0], "latent": ["84", 0], "strength": 1.0, "bypass": False}},
        "83": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["108", 0], "audio_latent": ["80", 1]}},
        "82": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["81", 0], "negative": ["81", 1], "cfg": 1.0}},
        "67": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "66": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "gradient_estimation"}},
        "73": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.909375, 0.725, 0.421875, 0.0"}},
        "70": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["67", 0], "guider": ["82", 0], "sampler": ["66", 0], "sigmas": ["73", 0], "latent_image": ["83", 0]}},
        "105": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["70", 1]}},
        "95": {"class_type": "VAEDecode", "inputs": {"samples": ["105", 0], "vae": ["1", 2]}},
        "200": {"class_type": "SaveImage", "inputs": {"images": ["95", 0], "filename_prefix": "ltx2"}},
    }
    wf["1"]["inputs"]["ckpt_name"] = wf["48"]["inputs"]["ckpt_name"] = wf["60"]["inputs"]["ckpt_name"] = ckpt
    if pin > 0:
        g = lambda pos, neg, lat: {"class_type": "LTXVAddGuide", "inputs": {
            "positive": pos, "negative": neg, "vae": ["1", 2], "latent": lat, "image": end,
            "frame_idx": -1, "strength": pin}}
        end = ["99", 0]
        if end_name:
            wf["98e"] = {"class_type": "LoadImage", "inputs": {"image": end_name}}
            wf["99e"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["98e", 0], "img_compression": 33}}
            end = ["99e", 0]
        wf["120"] = g(["22", 0], ["22", 1], ["107", 0])          # pass 1: last frame = the still
        wf["56"]["inputs"]["video_latent"] = ["120", 2]
        wf["47"]["inputs"].update(positive=["120", 0], negative=["120", 1])
        wf["81"]["inputs"].update(positive=["120", 0], negative=["120", 1])
        wf["121"] = g(["81", 0], ["81", 1], ["108", 0])          # pass 2: same pin after the upscale
        wf["83"]["inputs"]["video_latent"] = ["121", 2]
        wf["82"]["inputs"].update(positive=["121", 0], negative=["121", 1])
        wf["122"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": ["121", 0], "negative": ["121", 1], "latent": ["105", 0]}}
        wf["95"]["inputs"]["samples"] = ["122", 2]
    return wf


def run(still, out, prompt, w=704, h=1280, frames=121, pin=0.0, seed=None, ckpt=CKPT, end_still=None):
    fdir = os.path.join(out, "frames_ltx2")
    os.makedirs(fdir, exist_ok=True)
    sized = os.path.join(out, "_input_%dx%d.png" % (w, h))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", still, "-vf",
                    f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", sized], check=True)
    end_name = None
    if end_still:
        end_sized = os.path.join(out, "_end_%dx%d.png" % (w, h))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", end_still, "-vf",
                        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", end_sized], check=True)
        end_name = comfy.upload_image(end_sized)
    pid = comfy.submit(workflow(comfy.upload_image(sized), prompt, w, h, frames, seed, pin, ckpt, end_name))
    print("submitted", pid)
    comfy.wait(pid, timeout=1800)
    files = sorted(comfy.outputs(pid), key=lambda f: f.get("display_name", f["filename"]))
    for i, f in enumerate(files):
        comfy.download(f, os.path.join(fdir, f"f{i:04d}.png"))
    mp4 = os.path.join(out, "scene_ltx2.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(fdir, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    print("frames", len(files), "saved", mp4)
    return mp4


if __name__ == "__main__":
    # STILL OUTDIR PROMPT [W H FRAMES [PIN]]
    a = sys.argv
    run(a[1], a[2], a[3], *[int(x) for x in a[4:7]], *[float(x) for x in a[7:8]])
