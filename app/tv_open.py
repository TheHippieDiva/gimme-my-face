"""Rabbit-ears TV opening: static -> rolling VHS bar -> the picture tunes in -> the camera pushes THROUGH the screen to full frame.

    python3 app/tv_open.py TV.png INNER.mp4 SONG.mp3 OUT.mp4 [DUR]

TV.png has a solid bright-green screen (keyed here, on this computer). INNER.mp4 plays inside the screen.
"""
import subprocess, sys
import numpy as np
from PIL import Image

W, H, FPS = 1080, 1920, 30


def frames(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", path, "-vf", f"scale={W}:{H},fps={FPS}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def main(tv, inner, song, out, dur=4.5):
    t = Image.open(tv).convert("RGB")
    s = max(W / t.width, H / t.height); t = t.resize((int(t.width * s), int(t.height * s)), Image.LANCZOS)
    t = t.crop(((t.width - W) // 2, (t.height - H) // 2, (t.width - W) // 2 + W, (t.height - H) // 2 + H))
    a = np.asarray(t).astype(np.float32)
    green = (a[..., 1] > 170) & (a[..., 0] < 140) & (a[..., 2] < 140)
    ys, xs = np.nonzero(green)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    zfull = min(W / (x1 - x0), H / (y1 - y0)) * 1.0      # zoom that fills the frame with the screen
    m = np.asarray(Image.fromarray((green * 255).astype(np.uint8)).resize((W, H)), float)[..., None] / 255
    rng = np.random.default_rng(3)
    src = frames(inner)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-ss", "0", "-t", str(dur), "-i", song, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                            "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)
    n = int(dur * FPS)
    for i in range(n):
        tt = i / FPS
        pic = next(src).astype(np.float32)
        # what the screen shows: snow -> rolling, wobbly picture -> clean picture
        tune = np.clip((tt - 1.0) / 1.2, 0, 1)
        snow = rng.integers(0, 255, (H, W, 1)).repeat(3, 2).astype(np.float32)
        if tune < 1:
            roll = int((1 - tune) * H * ((tt * 1.3) % 1))
            pic = np.roll(pic, roll, 0)
            pic[:, :, 0] = np.roll(pic[:, :, 0], int(14 * (1 - tune)), 1)
        # the picture is fitted INTO the screen box, so the push-in reveals it naturally
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        sc = max(bw / W, bh / H)
        fit = np.asarray(Image.fromarray(np.clip(pic, 0, 255).astype(np.uint8)).resize((int(W * sc) + 1, int(H * sc) + 1), Image.BILINEAR)).astype(np.float32)
        fx0, fy0 = (fit.shape[1] - bw) // 2, (fit.shape[0] - bh) // 2
        boxed = np.zeros_like(pic); boxed[y0:y1 + 1, x0:x1 + 1] = fit[fy0:fy0 + bh, fx0:fx0 + bw]
        screen = snow * (1 - tune) + boxed * tune
        screen[::4] *= 0.85
        frame = a * (1 - m) + screen * m
        # push through the screen from 2.6 s
        p = np.clip((tt - 2.6) / 1.4, 0, 1); p = p * p * (3 - 2 * p)
        z = 1 + (zfull - 1) * p
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        cw, ch = W / z, H / z
        ccx = W / 2 + (cx - W / 2) * p; ccy = H / 2 + (cy - H / 2) * p   # frame centre -> screen centre
        ox = min(max(ccx - cw / 2, 0), W - cw); oy = min(max(ccy - ch / 2, 0), H - ch)
        img = img.crop((int(ox), int(oy), int(ox + cw), int(oy + ch))).resize((W, H), Image.BILINEAR)
        if p > 0.75:                                      # dissolve into the full-frame picture
            k = (p - 0.75) / 0.25
            img = Image.fromarray(np.clip(np.asarray(img, float) * (1 - k) + pic * k, 0, 255).astype(np.uint8))
        enc.stdin.write(np.asarray(img).tobytes())
    enc.stdin.close(); enc.wait()
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    print(main(a[0], a[1], a[2], a[3], float(a[4]) if len(a) > 4 else 4.5))
