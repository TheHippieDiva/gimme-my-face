"""YOUR MOVES, AI-YOU: Wan 2.2 Animate (open weights, on Comfy Cloud) makes a character picture do the dance in your video.

    python3 app/animate_dance.py DANCE.mp4 CHARACTER.png OUT.mp4 [--start S]

Move mode (pose transfer): your body + hand pose and your face motion are read from the video (DWPose), and the character
in CHARACTER.png performs them. One run = 77 frames at 16 fps (~4.8 s). Output keeps the character's own look.
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402

W, H, LEN, FPS = 480, 832, 77, 16


def workflow(video_name, image_name, prompt, seed=None):
    dw = lambda hand, body, face: {"class_type": "DWPreprocessor", "inputs": {
        "image": ["sc", 0], "detect_hand": hand, "detect_body": body, "detect_face": face, "resolution": 512,
        "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable"}}
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
        "l1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "l2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["l1", 0], "lora_name": "wan2.2_animate_14B_relight_lora_bf16.safetensors", "strength_model": 1.0}},
        "ms": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["l2", 0], "shift": 8.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "cv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "ref": {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
        "cve": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["cv", 0], "image": ["ref", 0], "crop": "none"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": "blurry, static, distorted face, extra fingers, bad hands, text, watermark"}},
        "vid": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "comp": {"class_type": "GetVideoComponents", "inputs": {"video": ["vid", 0]}},
        "sc": {"class_type": "ImageScale", "inputs": {"image": ["comp", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
        "pose": dw("enable", "enable", "disable"),
        "face": dw("disable", "disable", "enable"),
        "wa": {"class_type": "WanAnimateToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": W, "height": H, "length": LEN,
            "batch_size": 1, "continue_motion_max_frames": 5, "video_frame_offset": 0, "clip_vision_output": ["cve", 0],
            "reference_image": ["ref", 0], "pose_video": ["pose", 0], "face_video": ["face", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "positive": ["wa", 0], "negative": ["wa", 1], "latent_image": ["wa", 2],
               "seed": seed if seed is not None else random.randint(0, 2**31), "steps": 6, "cfg": 1.0,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "trim": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["ks", 0], "trim_amount": ["wa", 3]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["trim", 0], "vae": ["vae", 0]}},
        "fb": {"class_type": "ImageFromBatch", "inputs": {"image": ["dec", 0], "batch_index": ["wa", 4], "length": 4096}},
        "cvid": {"class_type": "CreateVideo", "inputs": {"images": ["fb", 0], "fps": float(FPS)}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["cvid", 0], "filename_prefix": "video/dance", "format": "auto", "format.codec": "auto"}},
    }


def run_pose(pose_video, character, out, prompt="The woman dances joyfully, full of energy, same outfit, same face, her locs swinging."):
    """Drive the character from a ready-made OpenPose stick-figure video (made on-device, so no other people's pixels
    ever leave the computer). No face video."""
    wf = workflow(comfy.upload_image(pose_video), comfy.upload_image(character), prompt)
    for k in ("sc", "pose", "face"):
        wf.pop(k)
    wf["psc"] = {"class_type": "ImageScale", "inputs": {"image": ["comp", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}}
    wf["wa"]["inputs"]["pose_video"] = ["psc", 0]
    wf["wa"]["inputs"].pop("face_video")
    pid = comfy.submit(wf)
    print("submitted", pid)
    comfy.wait(pid, timeout=2400)
    return comfy.download(comfy.outputs(pid)[0], out)


def run(dance, character, out, start=0.0, prompt="The woman dances joyfully, full of energy, same outfit, same face."):
    tmp = out.replace(".mp4", "_in.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", dance, "-t", str(LEN / FPS + 0.2),
                    "-vf", f"fps={FPS}", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", tmp], check=True)
    pid = comfy.submit(workflow(comfy.upload_image(tmp), comfy.upload_image(character), prompt))
    print("submitted", pid)
    comfy.wait(pid, timeout=2400)
    return comfy.download(comfy.outputs(pid)[0], out)


if __name__ == "__main__":
    a = sys.argv[1:]
    s = float(a[a.index("--start") + 1]) if "--start" in a else 0.0
    print(run(a[0], a[1], a[2], s))
