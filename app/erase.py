"""ERASER (Dawn, 29 Oct 2026: '"artifact" removal = eraser' / "people understand erasing something").
Paint over a spot on a scene; this fills it in from the skin/background around it. Runs on your computer, free, instant.
    <facefusion venv python> app/erase.py SCENE.png MASK.png OUT.png     (mask: white = erase)
"""
import sys
import cv2
import numpy as np

src, mask_p, out = sys.argv[1:4]
img = cv2.imread(src)
raw = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
raw = cv2.resize(raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
if raw.ndim == 2:
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
mc = raw[..., :3]
# the page sends strokes on a TRANSPARENT background: alpha says where you painted; pure white strokes = erase,
# any other colour = paint that colour (works for black or grey hair too). Old masks (white on black) still erase.
painted = raw[..., 3] > 127 if raw.shape[2] == 4 else mc.max(axis=2) > 127
if painted.mean() > 0.6:      # an old-page mask (opaque black background): only the white strokes count (29 Oct #10 went grey)
    painted = mc.min(axis=2) > 200
cols = mc[painted].astype(np.float32)
if len(cols) and np.median(cols.min(axis=1)) < 245:
    # COLOUR BRUSH (Dawn 29 Oct: "match the hair color like photoshop where you click the source and then where you want to
    # correct the color"): strokes drawn in the picked colour -> recolour under them, keep the texture (shading + strands)
    tgt = cv2.cvtColor(np.uint8([[np.median(cols, axis=0)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]; mean_L = L[painted].mean()
    soft = cv2.GaussianBlur(painted.astype(np.float32), (0, 0), 3)[..., None]
    new = lab.copy()
    new[..., 0] = np.clip(L + (tgt[0] - mean_L) * 0.85, 0, 255)          # move brightness to the picked colour, keep detail
    new[..., 1] = lab[..., 1] * 0.15 + tgt[1] * 0.85
    new[..., 2] = lab[..., 2] * 0.15 + tgt[2] * 0.85
    out_lab = lab * (1 - soft) + new * soft
    cv2.imwrite(out, cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    print("recoloured ->", out)
    sys.exit()
m = painted.astype(np.uint8) * 255
m = cv2.dilate(m, np.ones((5, 5), np.uint8))                     # a little past the stroke, so no rim is left
fill = cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)
fill = cv2.inpaint(fill, m, 7, cv2.INPAINT_NS)                   # second pass smooths the TELEA streaks
# put the skin grain back (inpaint is too smooth): grain from the area around the spot
noise = (img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32))
ring = cv2.dilate(m, np.ones((31, 31), np.uint8)) - m
grain = noise[ring > 0].std(axis=0) if (ring > 0).any() else np.zeros(3)
n1 = cv2.GaussianBlur(np.random.normal(0, 1, fill.shape[:2]).astype(np.float32), (0, 0), 0.7)[..., None]   # one soft mono grain
fill = fill.astype(np.float32)
# match the fill to the SKIN around the spot, not to glasses frames / neon / hair at its edge (29 Oct: "magic eraser left a
# blue spot"): the warm, mid-bright pixels in a ring around the spot set the fill's colour and brightness
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
ring2 = (cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0) & (m == 0)
R = lab_img[ring2]
warm = R[(R[:, 1] > 134) & (R[:, 0] < 235) & (R[:, 2] > 128)]
skin = warm[warm[:, 0] >= np.percentile(warm[:, 0], 50)] if len(warm) else warm   # the brighter half: skin, not dark-red hair
if len(skin) > 30:
    f_lab = cv2.cvtColor(np.clip(fill, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = m > 0
    tgt = np.median(skin, axis=0)
    cur = np.median(f_lab[inside], axis=0)
    f_lab[..., 0] += (tgt[0] - cur[0]) * 0.8
    f_lab[..., 1] = f_lab[..., 1] * 0.25 + tgt[1] * 0.75
    f_lab[..., 2] = f_lab[..., 2] * 0.25 + tgt[2] * 0.75
    fill = cv2.cvtColor(np.clip(f_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
fill = cv2.GaussianBlur(fill, (0, 0), 1.2) + n1 * float(grain.mean()) * 0.25   # smooth the fill, then ONE light grain
soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2)[..., None]
res = img.astype(np.float32) * (1 - soft) + fill * soft
cv2.imwrite(out, np.clip(res, 0, 255).astype(np.uint8))
print("erased ->", out)
