"""Visualizer FX driven by the song: retro meets sci-fi. Runs on your computer (numpy + Pillow + ffmpeg), no GPU.

    python3 app/fx.py PICTURE.mp4 SONG.mp3 OUT.mp4 [--look neon85]

Every frame reacts to the music:
  bump       the picture punches in on every kick (low-end onsets), then eases back
  flash      brightness lift on the hits
  rgb split  chromatic split on the biggest hits
  vhs        scanlines + a touch of tape noise, always on
  tesseract  a neon 4D hypercube turning in the middle distance, spinning faster and glowing on the beat
  eq bars    an 80s stereo graphic-EQ along the bottom edge, from the real spectrum
Captions are NOT drawn here: assemble.py puts them on top afterwards, so they stay crisp and readable.
"""
import itertools
import math
import subprocess
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 22050
NEON = [(255, 61, 110), (63, 208, 201), (143, 107, 255), (255, 178, 63)]


def audio(song):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", song, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def envelopes(y, n_frames):
    """Per video frame: kick envelope (0..1), hit envelope, and a 24-band spectrum (0..1)."""
    hop = SR // FPS
    win = 2048
    pad = np.pad(y, (win, win + hop * n_frames))
    frames = np.stack([pad[win + i * hop - win // 2: win + i * hop + win // 2] for i in range(n_frames)])
    mag = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1 / SR)
    low = np.log1p(mag[:, (freqs > 30) & (freqs < 150)].sum(1))
    allb = np.log1p(mag.sum(1))

    def onset_env(x, decay):
        flux = np.maximum(0, np.diff(x, prepend=x[0]))
        flux = flux / (np.percentile(flux, 99) + 1e-9)
        env, e = np.zeros_like(flux), 0.0
        for i, f in enumerate(np.clip(flux, 0, 1)):
            e = max(f, e * decay)
            env[i] = e
        return env
    kick = onset_env(low, 0.80)
    hit = onset_env(allb, 0.75)
    edges = np.geomspace(40, 12000, 25)
    bands = np.stack([np.log1p(mag[:, (freqs >= a) & (freqs < b)].mean(1)) for a, b in zip(edges[:-1], edges[1:])], 1)
    bands = (bands - np.percentile(bands, 5, 0)) / (np.percentile(bands, 99, 0) - np.percentile(bands, 5, 0) + 1e-9)
    return kick, hit, np.clip(bands, 0, 1)


# --- tesseract: 16 vertices of a 4D cube, rotated in the XW and YZ planes, projected 4D -> 3D -> 2D
VERTS = np.array(list(itertools.product([-1, 1], repeat=4)), float)
EDGES = [(a, b) for a in range(16) for b in range(a + 1, 16) if np.sum(VERTS[a] != VERTS[b]) == 1]


def tesseract(angle, size, alpha, color):
    c, s = math.cos(angle), math.sin(angle)
    v = VERTS.copy()
    v[:, [0, 3]] = v[:, [0, 3]] @ np.array([[c, -s], [s, c]])
    c2, s2 = math.cos(angle * 0.6), math.sin(angle * 0.6)
    v[:, [1, 2]] = v[:, [1, 2]] @ np.array([[c2, -s2], [s2, c2]])
    w = 1 / (2.6 - v[:, 3])
    p3 = v[:, :3] * w[:, None]
    z = 1 / (2.8 - p3[:, 2])
    p2 = p3[:, :2] * z[:, None] * size + np.array([W / 2, H * 0.45])
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for a, b in EDGES:
        d.line([tuple(p2[a]), tuple(p2[b])], fill=color + (int(255 * alpha),), width=6)
    glow = layer.filter(ImageFilter.GaussianBlur(10))
    return Image.alpha_composite(glow, layer)


def eq_bars(bands, color_shift):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    n, gap = len(bands), 6
    bw = (W - 80 - gap * (n - 1)) / n
    for i, v in enumerate(bands):
        segs = int(v * 12)
        for k in range(segs):
            col = NEON[1] if k < 7 else NEON[3] if k < 10 else NEON[0]
            x0 = 40 + i * (bw + gap)
            y0 = H - 30 - (k + 1) * 20
            d.rectangle([x0, y0, x0 + bw, y0 + 14], fill=col + (210,))
    return layer


def main(picture, song, out, look="neon85"):
    from assemble import dur_of
    n = int(round(dur_of(picture) * FPS))
    kick, hit, bands = envelopes(audio(song), n)
    scan = np.ones((H, 1, 1), np.float32)
    scan[::3] = 0.86                                     # scanlines
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", picture, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    angle, rng = 0.0, np.random.default_rng(7)
    for i in range(n):
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        k, h = float(kick[min(i, len(kick) - 1)]), float(hit[min(i, len(hit) - 1)])
        img = Image.fromarray(f)
        z = 1 + 0.07 * k                                  # bump: punch in on the kick
        if z > 1.002:
            cw, ch = int(W / z), int(H / z)
            img = img.crop(((W - cw) // 2, (H - ch) // 2, (W - cw) // 2 + cw, (H - ch) // 2 + ch)).resize((W, H), Image.BILINEAR)
        a = np.asarray(img).astype(np.float32)
        a *= 1 + 0.35 * h                                 # flash
        if h > 0.6:                                       # rgb split on the big hits
            s = int(14 * (h - 0.5))
            a[:, :, 0] = np.roll(a[:, :, 0], s, 1)
            a[:, :, 2] = np.roll(a[:, :, 2], -s, 1)
        a = a * scan + rng.normal(0, 4, (H, 1, 1))        # vhs
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
        angle += 0.012 + 0.08 * k                         # the tesseract spins faster on the beat
        col = NEON[(i // (FPS * 4)) % 3]
        img = Image.alpha_composite(img, tesseract(angle, 2700 + 700 * k, 0.22 + 0.55 * h, col))
        img = Image.alpha_composite(img, eq_bars(bands[min(i, len(bands) - 1)], i))
        enc.stdin.write(img.convert("RGB").tobytes())
        if i % 300 == 0:
            print(f"fx {i}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*a[:3])


def inside(bg_path, person_path, song, out, t0, dur, person_h=0.78):
    """YOU INSIDE THE TESSERACT: background world, the real you (cut-out PNG) facing front, and a neon hypercube that
    turns AROUND you: its far edges are drawn behind you, its near edges in front. Bumps + glow on the beat."""
    from PIL import ImageChops
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((W, int(bg.height * W / bg.width))) if bg.width / bg.height < W / H else bg.resize((int(bg.width * H / bg.height), H))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    me = Image.open(person_path).convert("RGBA")
    me = me.crop(me.getbbox())
    ph = int(H * person_h)
    me = me.resize((int(me.width * ph / me.height), ph), Image.LANCZOS)
    a = me.split()[3]                                   # orange rim light from the edge of the cut-out
    rim = Image.new("RGBA", me.size, (255, 150, 60, 0))
    rim.putalpha(ImageChops.subtract(a.filter(ImageFilter.GaussianBlur(9)), a.filter(ImageFilter.GaussianBlur(2))).point(lambda v: min(255, v * 3)))
    rgb = np.asarray(me).astype(np.float32)             # match the world's light: warmer, a touch darker
    rgb[:, :, :3] *= np.array([1.02, 0.86, 0.74]) * 0.92
    me = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGBA")
    px, py = (W - me.width) // 2, H + 30 - me.height      # run off the bottom edge: no cut-off look
    y = audio(song)[int(t0 * SR): int((t0 + dur) * SR)]
    n = int(dur * FPS)
    kick, hit, bands = envelopes(y, n)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    angle = 0.0
    for i in range(n):
        k, h = float(kick[i]), float(hit[i])
        angle += 0.01 + 0.07 * k
        frame = bg.copy().convert("RGBA")
        back, front = cube_layers(angle, 1500 + 300 * k, 0.35 + 0.5 * h, NEON[(i // (FPS * 4)) % 3])
        frame = Image.alpha_composite(frame, back)
        lit = Image.alpha_composite(me, rim)
        frame.alpha_composite(lit, (px, py))
        frame = Image.alpha_composite(frame, front)
        frame = Image.alpha_composite(frame, eq_bars(bands[i], i))
        f = frame.convert("RGB")
        z = 1 + 0.02 * i / n + 0.06 * k                   # slow push-in + bump
        cw, ch = int(W / z), int(H / z)
        f = f.crop(((W - cw) // 2, (H - ch) // 2, (W - cw) // 2 + cw, (H - ch) // 2 + ch)).resize((W, H), Image.BILINEAR)
        arr = np.asarray(f).astype(np.float32) * (1 + 0.3 * h)
        if h > 0.6:
            s = int(12 * (h - 0.5))
            arr[:, :, 0] = np.roll(arr[:, :, 0], s, 1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -s, 1)
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def cube_layers(angle, size, alpha, color):
    """The hypercube split by depth: edges whose midpoint is far away (behind the person) and near (in front)."""
    c, s = math.cos(angle), math.sin(angle)
    v = VERTS.copy()
    v[:, [0, 3]] = v[:, [0, 3]] @ np.array([[c, -s], [s, c]])
    c2, s2 = math.cos(angle * 0.6), math.sin(angle * 0.6)
    v[:, [0, 2]] = v[:, [0, 2]] @ np.array([[c2, -s2], [s2, c2]])
    w = 1 / (2.6 - v[:, 3])
    p3 = v[:, :3] * w[:, None]
    z = 1 / (2.2 - p3[:, 2])
    p2 = p3[:, :2] * z[:, None] * size + np.array([W / 2, H * 0.52])
    layers = []
    for keep in (lambda d: d < 0, lambda d: d >= 0):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for a_, b_ in EDGES:
            if keep((p3[a_, 2] + p3[b_, 2]) / 2):
                d.line([tuple(p2[a_]), tuple(p2[b_])], fill=color + (int(255 * alpha),), width=7)
        layers.append(Image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(12)), layer))
    return layers


def _frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def polish(rgb, glam=False):
    """Classic, kind finishing on the real footage (no AI on the face): a little brighter, softer skin, warm.
    glam = more: smoother skin, a soft beauty glow, richer color."""
    im = Image.fromarray(rgb)
    im = Image.blend(im, im.filter(ImageFilter.GaussianBlur(2.2)), 0.42 if glam else 0.35)
    a = np.asarray(im).astype(np.float32) / 255
    a = a ** (0.85 if glam else 0.9) * 1.04               # lift the mids
    if glam:
        glow = np.asarray(Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255
        a = 1 - (1 - a) * (1 - 0.18 * glow)               # soft beauty glow (screen blend)
        g = a.mean(2, keepdims=True)
        a = g + (a - g) * 1.12                            # a touch richer color
        a = a * np.array([1.03, 1.0, 0.97])               # warm
    return np.clip(a * 255, 0, 255).astype(np.uint8)


def inside_video(bg_path, fg, mask, song, out, t0, dur, cut_y=1.0, person_h=0.80, src_size=None, glam=False, bottom=0.93):
    """Like inside(), but YOU MOVE: fg/mask videos from cutout.py. cut_y = keep the top fraction of the clip
    (e.g. just above the shirt neckline), with a soft fade at the cut."""
    from assemble import dur_of
    from PIL import ImageChops
    sw, sh = src_size
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((W, int(bg.height * W / bg.width))) if bg.width / bg.height < W / H else bg.resize((int(bg.width * H / bg.height), H))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    keep = int(sh * cut_y)
    ph = int(H * person_h)
    pw = int(sw * ph / keep)
    fade = np.ones((keep, 1), np.float32)
    f0 = max(1, int(keep * 0.08))
    fade[-f0:, 0] = np.linspace(1, 0, f0)                # soft edge at the neckline cut
    y = audio(song)[int(t0 * SR): int((t0 + dur) * SR)]
    n = int(dur * FPS)
    kick, hit, bands = envelopes(y, n)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    fgs, mks = _frames(fg, sw, sh), _frames(mask, sw, sh)
    angle, last = 0.0, None
    for i in range(n):
        try:
            fr, mk = next(fgs), next(mks)
            last = (fr, mk)
        except StopIteration:
            fr, mk = last                                 # clip shorter than the slot: hold the last frame
        k, h = float(kick[i]), float(hit[i])
        rgb = polish(fr[:keep].copy(), glam).astype(np.float32) * np.array([1.02, 0.88, 0.78])
        al = mk[:keep, :, 0].astype(np.float32) * fade
        me = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), al]).astype(np.uint8), "RGBA").resize((pw, ph), Image.LANCZOS)
        a = me.split()[3]
        rim = Image.new("RGBA", me.size, (255, 150, 60, 0))
        rim.putalpha(ImageChops.subtract(a.filter(ImageFilter.GaussianBlur(9)), a.filter(ImageFilter.GaussianBlur(2))).point(lambda v: min(255, v * 3)))
        angle += 0.01 + 0.07 * k
        frame = bg.copy().convert("RGBA")
        back, front = cube_layers(angle, 1500 + 300 * k, 0.35 + 0.5 * h, NEON[(i // (FPS * 4)) % 3])
        frame = Image.alpha_composite(frame, back)
        pos = ((W - pw) // 2, int(H * bottom) - ph)
        frame.alpha_composite(Image.alpha_composite(me, rim), pos)
        cover = Image.new("L", (W, H), 0)                 # near edges fade where they cross you (never across the face)
        cover.paste(me.split()[3].filter(ImageFilter.GaussianBlur(20)), pos)
        fa = np.asarray(front.split()[3]).astype(np.float32) * (1 - 0.85 * np.asarray(cover).astype(np.float32) / 255)
        front.putalpha(Image.fromarray(fa.astype(np.uint8)))
        frame = Image.alpha_composite(frame, front)
        frame = Image.alpha_composite(frame, eq_bars(bands[i], i))
        f = frame.convert("RGB")
        z = 1 + 0.02 * i / n + 0.05 * k
        cw, ch = int(W / z), int(H / z)
        f = f.crop(((W - cw) // 2, (H - ch) // 2, (W - cw) // 2 + cw, (H - ch) // 2 + ch)).resize((W, H), Image.BILINEAR)
        arr = np.asarray(f).astype(np.float32) * (1 + 0.3 * h)
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def cgi_skin(rgb):
    """1985 computer-host look on REAL footage: plastic-smooth, contrasty, hot specular highlights, a few color steps."""
    a = polish(rgb).astype(np.float32) / 255 * 0.92
    sm = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))).astype(np.float32) / 255
    a = 0.4 * a + 0.6 * sm                                 # plastic-smooth (edges come back from the contrast below)
    a = np.clip((a - 0.5) * 1.2 + 0.48, 0, 1)             # punchy contrast
    spec = np.clip((a.mean(2, keepdims=True) - 0.78) * 3.0, 0, 1)
    a = np.clip(a + 0.25 * spec, 0, 1)                    # glossy hot spots
    a = np.round(a * 20) / 20                              # a few color steps, like early rendering
    return (a * 255).astype(np.uint8)


def headroom(bg_path, fg, mask, song, out, t0, dur, cut_y=0.52, person_h=0.66, src_size=None, bottom=0.86):
    """The 80s computer-TV-host treatment on YOUR clip: spinning neon tunnel, glossy CGI skin, stutter-jumps on the beat."""
    from PIL import ImageChops
    sw, sh = src_size
    tun = Image.open(bg_path).convert("RGB").resize((int(H * 1.25), int(H * 1.25)))
    keep = int(sh * cut_y)
    ph = int(H * person_h)
    pw = int(sw * ph / keep)
    fade = np.ones((keep, 1), np.float32)
    f0 = max(1, int(keep * 0.10))
    fade[-f0:, 0] = np.linspace(1, 0, f0)
    y = audio(song)[int(t0 * SR): int((t0 + dur) * SR)]
    n = int(dur * FPS)
    kick, hit, bands = envelopes(y, n)
    frames = list(zip(_frames(fg, sw, sh), _frames(mask, sw, sh)))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    rng = np.random.default_rng(3)
    ang, stutter = 0.0, 0
    for i in range(n):
        k, h = float(kick[i]), float(hit[i])
        if h > 0.75 and stutter == 0 and rng.random() < 0.6:
            stutter = 6                                   # the famous st-st-stutter: a few frames bounce back and forth
        j = min(i, len(frames) - 1)
        if stutter:
            j = max(0, j - (2 if stutter % 2 else 0))
            stutter -= 1
        fr, mk = frames[j]
        ang += 0.15 + 2.5 * k                             # the backdrop spins, kicked by the beat
        bg = tun.rotate(ang, resample=Image.BILINEAR)
        cx, cy = bg.width // 2, bg.height // 2
        frame = bg.crop((cx - W // 2, cy - H // 2, cx + W // 2, cy + H // 2)).convert("RGBA")
        rgb = cgi_skin(fr[:keep].copy())
        al = mk[:keep, :, 0].astype(np.float32) * fade
        me = Image.fromarray(np.dstack([rgb, al.astype(np.uint8)]), "RGBA")
        jump = 1.0 + (0.06 if stutter else 0) + 0.03 * k
        me = me.resize((int(pw * jump), int(ph * jump)), Image.LANCZOS)
        a = me.split()[3]
        rim = Image.new("RGBA", me.size, (120, 240, 255, 0))
        rim.putalpha(ImageChops.subtract(a.filter(ImageFilter.GaussianBlur(8)), a.filter(ImageFilter.GaussianBlur(2))).point(lambda v: min(255, v * 3)))
        dx = int(rng.normal(0, 14)) if stutter else 0
        frame.alpha_composite(Image.alpha_composite(me, rim), ((W - me.width) // 2 + dx, int(H * bottom) - me.height))
        arr = np.asarray(frame.convert("RGB")).astype(np.float32)
        if h > 0.55 or stutter:
            s = int(10 + 14 * h)
            arr[:, :, 0] = np.roll(arr[:, :, 0], s, 1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -s, 1)
        arr[::3] *= 0.85                                  # scanlines
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def pencil(rgb):
    """Pencil-sketch look from a colour frame (colour dodge + paper grain): same face, never a different drawing."""
    from PIL import ImageOps
    g = Image.fromarray(rgb).convert("L")
    b = np.asarray(ImageOps.invert(g).filter(ImageFilter.GaussianBlur(6)), float)
    a = np.asarray(g, float)
    p = np.clip(a * 255 / (256 - b), 0, 255)
    return np.dstack([p, p * 0.98, p * 0.94]).astype(np.uint8)


def recolor(rgb, shift=0.45):
    """Same drawing, different colour scheme (hue rotation, punchier)."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.int16)
    skin = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 28) & (hsv[..., 1] > 40) & (hsv[..., 1] < 170) & (hsv[..., 2] > 110)
    hsv[..., 0] = np.where(skin, hsv[..., 0], (hsv[..., 0] + int(shift * 255)) % 256)
    hsv[..., 1] = np.where(skin, hsv[..., 1], np.clip(hsv[..., 1] * 1.25, 0, 255))
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def ghost(rgb, alpha, color):
    """Glowing outline ghost: the silhouette edge + the drawing's own ink lines, in neon, no fill."""
    a = Image.fromarray(alpha).convert("L")
    rim = np.asarray(a.filter(ImageFilter.MaxFilter(5)), float) - np.asarray(a.filter(ImageFilter.MinFilter(5)), float)
    g = Image.fromarray(rgb).convert("L")
    edges = np.asarray(g.filter(ImageFilter.FIND_EDGES), float)
    ink = (edges > 70) & (np.asarray(a.filter(ImageFilter.MinFilter(9)), float) > 128)
    line = np.clip(rim / 255 + ink * 0.7, 0, 1)
    L = Image.fromarray((line * 255).astype(np.uint8))
    glow = np.asarray(L.filter(ImageFilter.GaussianBlur(6)), float) / 255
    al = np.clip(line + glow * 0.7, 0, 1) * 0.9
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(np.array(color) * (0.6 + 0.4 * line[..., None]) + 255 * line[..., None] * 0.35, 0, 255)
    out[..., 3] = (al * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def trio(bg_path, fg, mask, song, out, t0, dur, src_size, lag=3):
    """Three of you inside the tesseract: colour in front, pencil sketch left, alt colour right (a few frames behind = canon)."""
    sw, sh = src_size
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((W, int(bg.height * W / bg.width))) if bg.width / bg.height < W / H else bg.resize((int(bg.width * H / bg.height), H))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    frames = list(zip(_frames(fg, sw, sh), _frames(mask, sw, sh)))
    y = audio(song)[int(t0 * SR): int((t0 + dur) * SR)]
    n = int(dur * FPS)
    kick, hit, bands = envelopes(y, n)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    angle = 0.0
    def sprite(j, kind, h):
        fr, mk = frames[max(0, min(j, len(frames) - 1))]
        if kind == "c":
            im = Image.fromarray(np.dstack([fr, mk[:, :, 0]]), "RGBA")
        else:
            im = ghost(fr, mk[:, :, 0], (255, 61, 160) if kind == "s" else (63, 220, 255))
        return im.resize((int(sw * h / sh), h), Image.LANCZOS)
    for i in range(n):
        k, hh = float(kick[i]), float(hit[i])
        j = int(i * len(frames) / n)
        angle += 0.01 + 0.07 * k
        frame = bg.copy().convert("RGBA")
        back, front = cube_layers(angle, 1500 + 300 * k, 0.35 + 0.5 * hh, NEON[(i // (FPS * 4)) % 3])
        frame = Image.alpha_composite(frame, back)
        side_h, mid_h = int(H * 0.66), int(H * 0.72)
        for kind, cx, h, lagf in (("s", 0.24, side_h, lag), ("a", 0.76, side_h, 2 * lag), ("c", 0.5, mid_h, 0)):
            sp = sprite(j - lagf, kind, h)
            frame.alpha_composite(sp, (int(W * cx) - sp.width // 2, int(H * 0.96) - sp.height))
        frame = Image.alpha_composite(frame, eq_bars(bands[i], i))
        f = frame.convert("RGB")
        z = 1 + 0.05 * k
        cw, ch = int(W / z), int(H / z)
        f = f.crop(((W - cw) // 2, (H - ch) // 2, (W - cw) // 2 + cw, (H - ch) // 2 + ch)).resize((W, H), Image.BILINEAR)
        arr = np.asarray(f).astype(np.float32) * (1 + 0.25 * hh)
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return out
