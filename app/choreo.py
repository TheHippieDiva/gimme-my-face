"""Beat-locked choreography as OpenPose stick figures (the motion input for Wan Animate). No camera, no people needed.

    python3 app/choreo.py OUT.mp4 BPM [real_pose.json]

Moves (Gen X club classics, as body mechanics): 'wop' (shoulders + elbows bounce side to side, knees pump),
'chacha' (rock step back, triple step, hips sway), 'hiproll' (cheeky hip circles, arms up), 'real' (your own tracked moves,
re-centred and held at one size so you never walk off screen).
"""
import json, math, sys
import numpy as np
from PIL import Image, ImageDraw
import subprocess, os

W, H, FPS = 480, 832, 16
LIMBS = [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
COL = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]
# neutral standing pose (OpenPose-18, pixels in a 480x832 frame)
N = {0:(240,95),1:(240,165),2:(195,170),3:(178,265),4:(170,355),5:(285,170),6:(302,265),7:(310,355),
     8:(212,400),9:(208,560),10:(205,720),11:(268,400),12:(272,560),13:(275,720),14:(228,85),15:(252,85),16:(214,92),17:(266,92)}


def draw(P, sx=0.78, sy=0.82, oy=70):
    # slimmer + smaller in frame, room above the hands and below the feet
    P = {k: (240 + (x - 240) * sx, 400 + (y - 400) * sy + oy) for k, (x, y) in P.items()}
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    for li, (a, b) in enumerate(LIMBS):
        if a in P and b in P: d.line([P[a], P[b]], fill=COL[li], width=7)
    for i, (x, y) in P.items(): d.ellipse([x-5, y-5, x+5, y+5], fill=COL[i])
    return im


def move(name, ph):
    """ph = phase in beats (float). Returns a pose dict."""
    P = {k: list(v) for k, v in N.items()}
    s = math.sin(ph * math.pi)            # one swing per beat
    s2 = math.sin(ph * math.pi / 2)       # half speed
    def shift(keys, dx, dy):
        for k in keys: P[k][0] += dx; P[k][1] += dy
    up = [0,1,2,3,4,5,6,7,14,15,16,17]
    if name == 'wop':                      # bounce + shoulders side to side, elbows bent and pumping
        b = abs(s) * 18
        shift(up + [8, 11], 22 * s2, b)
        P[3] = [P[2][0] - 40, P[2][1] + 60]; P[4] = [P[3][0] + 25 + 25 * s, P[3][1] - 40]
        P[6] = [P[5][0] + 40, P[5][1] + 60]; P[7] = [P[6][0] - 25 + 25 * s, P[6][1] - 40]
        P[9][0] -= 10; P[12][0] += 10; P[9][1] += b * 0.6; P[12][1] += b * 0.6
    elif name == 'chacha':                 # rock back on 2-3, triple step 4&1, hips sway
        cyc = (ph % 4) / 4
        back = 40 * math.sin(cyc * 2 * math.pi)
        sway = 25 * math.sin(ph * math.pi)
        shift(up, sway * 0.4, 0); shift([8, 11], sway, 0)
        P[13][1] -= max(0, back) * 0.5; P[12][0] += sway * 0.5; P[10][0] += sway * 0.3
        P[4] = [P[3][0] - 30, P[3][1] - 10 + 15 * s]; P[7] = [P[6][0] + 30, P[6][1] - 10 - 15 * s]
    elif name == 'hiproll':                # arms up, hips circle
        cx = 30 * math.cos(ph * math.pi / 2); cy = 14 * math.sin(ph * math.pi / 2)
        shift([8, 11], cx, cy); shift([9, 12], cx * 0.6, 0)
        P[3] = [P[2][0] - 20, P[2][1] - 70]; P[4] = [P[3][0] + 15, P[3][1] - 80]
        P[6] = [P[5][0] + 20, P[5][1] - 70]; P[7] = [P[6][0] - 15, P[6][1] - 80]
        shift([0, 14, 15, 16, 17], -cx * 0.3, 0)
    elif name == 'jungle':                 # arm pumps overhead, left-right, on every beat
        side = 1 if int(ph) % 2 == 0 else -1
        P[3] = [P[2][0] + 10 * side, P[2][1] - 60]; P[4] = [P[3][0] + 35 * side, P[3][1] - 70 - 20 * abs(s)]
        P[6] = [P[5][0] + 10 * side, P[5][1] - 60]; P[7] = [P[6][0] + 35 * side, P[6][1] - 70 - 20 * abs(s)]
        shift(up, 15 * side * abs(s), 0); shift([8, 11], 20 * side * abs(s), 0)
    A = 1.8                                # bigger, clearer moves
    Q = {k: [N[k][0] + A * (v[0] - N[k][0]), N[k][1] + A * (v[1] - N[k][1])] for k, v in P.items()}
    # keep every bone at its true length (only the angles move): no long arms, no stretched legs
    for a_, b_ in ((2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13)):
        if a_ in Q and b_ in Q and a_ in N and b_ in N:
            L = math.dist(N[a_], N[b_]); dx, dy = Q[b_][0] - Q[a_][0], Q[b_][1] - Q[a_][1]
            d = math.hypot(dx, dy) or 1.0
            nb = [Q[a_][0] + dx / d * L, Q[a_][1] + dy / d * L]
            off = (nb[0] - Q[b_][0], nb[1] - Q[b_][1])
            for c in ({3: [4], 6: [7], 9: [10], 12: [13]}.get(b_, [])):
                Q[c][0] += off[0]; Q[c][1] += off[1]
            Q[b_] = nb
    return {k: tuple(v) for k, v in Q.items()}


def real_frames(path):
    """Your tracked moves (Vision joints) re-centred on the hips and held at one size."""
    OP = ['head_joint','neck_1_joint','right_shoulder_1_joint','right_forearm_joint','right_hand_joint','left_shoulder_1_joint','left_forearm_joint','left_hand_joint',
          'right_upLeg_joint','right_leg_joint','right_foot_joint','left_upLeg_joint','left_leg_joint','left_foot_joint','right_eye_joint','left_eye_joint','right_ear_joint','left_ear_joint']
    out = []
    for p in json.load(open(path)):
        pts = {i: (p[n][0], 1 - p[n][1]) for i, n in enumerate(OP) if n in p}
        if 1 not in pts or 8 not in pts or 11 not in pts: out.append(None); continue
        hip = ((pts[8][0] + pts[11][0]) / 2, (pts[8][1] + pts[11][1]) / 2)
        torso = math.dist(pts[1], hip) or 1e-3
        sc = 0.85 * (N[8][1] - N[1][1]) / torso                    # torso length fixed = no walking toward the camera
        out.append({i: ((x - hip[0]) * sc + 240, (y - hip[1]) * sc + 400) for i, (x, y) in pts.items()})
    last = None
    for i, f in enumerate(out):
        if f is None: out[i] = last
        else: last = f
    return [f for f in out if f]


def neutral_from_ref(pose_jsonl):
    """The character's OWN skeleton (Vision pose on the reference image), in the 480x832 frame Wan Animate uses:
    every move keeps her drawn proportions, so nothing gets stretched or squashed."""
    OP = ['head_joint','neck_1_joint','right_shoulder_1_joint','right_forearm_joint','right_hand_joint','left_shoulder_1_joint','left_forearm_joint','left_hand_joint',
          'right_upLeg_joint','right_leg_joint','right_foot_joint','left_upLeg_joint','left_leg_joint','left_foot_joint','right_eye_joint','left_eye_joint','right_ear_joint','left_ear_joint']
    p = json.loads(open(pose_jsonl).readline())['people'][0]
    return {i: (p[n][0] * W, (1 - p[n][1]) * H) for i, n in enumerate(OP) if n in p}


def neutral_from(real):
    """Use YOUR body proportions for every made-up move (average of your tracked, re-centred frames)."""
    rf = real_frames(real)
    keys = set.intersection(*[set(f) for f in rf])
    n = {k: (float(np.mean([f[k][0] for f in rf])), float(np.mean([f[k][1] for f in rf]))) for k in keys}
    # stand tall: straight legs, feet under the hips, legs as long as the body needs (no squat)
    for hip, knee, foot in ((8, 9, 10), (11, 12, 13)):
        if hip in n and foot in n:
            hx, hy = n[hip]; leg = 1.4 * (n[8][1] - n[1][1]) if 8 in n and 1 in n else 300
            n[knee] = (hx + (-6 if hip == 8 else 6), hy + leg / 2)
            n[foot] = (hx + (-10 if hip == 8 else 10), hy + leg)
    return n


def main(out, bpm, real=None, ref_pose=None, frame=(0.78, 0.82, 70), plan=(('real', 2.5), ('wop', 4), ('chacha', 4), ('hiproll', 4), ('jungle', 4))):
    spb = 60 / bpm; frames = []
    rf = real_frames(real) if real else []
    if ref_pose:
        N.update(neutral_from_ref(ref_pose))
    elif real:
        N.update(neutral_from(real))
    for name, secs_or_beats in plan:
        if name == 'real':
            frames += rf[:int(secs_or_beats * FPS)]
        else:
            n = int(secs_or_beats * spb * FPS)
            frames += [move(name, k / FPS / spb) for k in range(n)]
    d = out + '_frames'; os.makedirs(d, exist_ok=True)
    for k, P in enumerate(frames): draw(P, *frame).save(f'{d}/p{k:04d}.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', f'{d}/p%04d.png', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', out], check=True)
    return out, len(frames) / FPS


if __name__ == '__main__':
    print(main(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None))
