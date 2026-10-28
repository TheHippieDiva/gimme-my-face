"""Assemble the music video from a SLOT LIST: one slot per storyboard scene, every cut on the beat.

    python3 app/assemble.py SONG.mp3 OUT_DIR SLOTS.json [--tags] [--from S --to S]

OUT_DIR holds timeline.json + captions.ass from lyricbeat.py. SLOTS.json lists the slots in order:
    {"id": "05", "at": 23.18, "what": "laptop at the kitchen table", "media": ["out/.../a.png", "clip.mp4"],
     "status": "placeholder" | "ai" | "real", "cut": 2}
`at` = when the slot starts (snapped to the nearest bar); a slot runs until the next one starts.
Inside a slot the picture changes every `cut` bars (default 2), cycling through `media`: stills get a slow push
or drift (alternating), clips play (slowed up to 2x to fit, then hold their last frame).
Swap a shot = change its `media` and run again. --tags puts the slot number in a corner (review copies only).
Captions are added LAST, on top of the finished picture; nothing with text ever goes back into a model.
"""
import json
import os
import subprocess
import sys

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dur_of(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    h, m, s = err.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def snap(t, grid):
    return min(grid, key=lambda g: abs(g - t))


def shot(src, d, k, path):
    """One shot of d seconds from a still or a clip, scaled/cropped to fill 1080x1920."""
    n = max(1, round(d * FPS))
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if src.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        # big canvas first so the slow push doesn't shimmer; 4 moves, alternating per shot
        z = ["1.0+0.10*on/{n}", "1.10-0.10*on/{n}", "1.06", "1.06"][k % 4].format(n=n)
        x = ["iw/2-(iw/zoom/2)", "iw/2-(iw/zoom/2)", "(iw-iw/zoom)*on/{n}", "(iw-iw/zoom)*(1-on/{n})"][k % 4].format(n=n)
        vf = (f"scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,crop={2 * W}:{2 * H},"
              f"zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}")
        cmd = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", src]
    else:
        cd = dur_of(src)
        slow = min(2.0, max(1.0, d / cd))              # slow a short clip down up to 2x, then hold the last frame
        vf = f"{fill},setpts={slow:.4f}*PTS,fps={FPS},tpad=stop_mode=clone:stop_duration={d:.3f}"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src]
    cmd += ["-vf", vf + ",format=yuv420p", "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-r", str(FPS), path]
    subprocess.run(cmd, check=True)


def tag_events(slots, starts, ends):
    """Slot numbers as an ASS layer (review copies only)."""
    from lyricbeat import ass_time
    return [f"Dialogue: 1,{ass_time(s)},{ass_time(e)},Tag,,0,0,0,,#{sl['id']}" + (" STILL" if sl.get("status") == "placeholder" else "")
            for sl, s, e in zip(slots, starts, ends)]


def main(song, out, slots_path, tags=False, t_from=0.0, t_to=None):
    sys.path.insert(0, os.path.dirname(__file__))
    tl = json.load(open(f"{out}/timeline.json"))
    spec = json.load(open(slots_path))
    slots = [sl for sl in spec["slots"] if sl.get("media")]   # a slot with no picture yet: the one before it runs on
    if not slots:
        raise SystemExit("no slot has a picture yet")
    total = dur_of(song)
    t_to = t_to or total
    bars = [0.0] + tl["beats"][::4][1:] + [total]
    starts = [0.0] + [snap(sl["at"], bars) for sl in slots[1:]]
    ends = starts[1:] + [total]
    work = os.path.join(out, "shots")
    os.makedirs(work, exist_ok=True)
    listing, made = [], 0
    for si, (sl, s, e) in enumerate(zip(slots, starts, ends)):
        if e <= t_from or s >= t_to:
            continue
        grid = [b for b in bars if s <= b < e] + [e]
        step = sl.get("cut", 2)
        cuts = grid[::step] + ([e] if grid[::step][-1] != e else [])
        media = [m if os.path.isabs(m) else os.path.join(ROOT, m) for m in sl["media"]]
        for k in range(len(cuts) - 1):
            p = os.path.join(work, f"s{sl['id']}_{k:02d}.mp4")
            shot(media[k % len(media)], cuts[k + 1] - cuts[k], si + k, p)
            listing.append(f"file '{os.path.abspath(p)}'")
            made += 1
    lst = os.path.join(work, "list.txt")
    open(lst, "w").write("\n".join(listing) + "\n")
    picture = os.path.join(out, "picture.mp4")       # clean picture, no captions (reusable)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", picture], check=True)
    # captions (+ end card, + optional slot tags) as ONE top layer
    ass = open(f"{out}/captions.ass", encoding="utf-8").read()
    style_tag = "Style: Tag,Arial,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,40,40,36,1"
    style_end = "Style: End,Georgia,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,1,0,0,100,100,2,0,1,3,2,5,60,60,0,1"
    ass = ass.replace("\n[Events]", f"\n{style_tag}\n{style_end}\n\n[Events]".replace("\n\n\n", "\n\n"))
    from lyricbeat import ass_time
    card = spec.get("end_card")
    extra = []
    if card:
        extra.append(f"Dialogue: 2,{ass_time(total - card.get('secs', 4))},{ass_time(total)},End,,0,0,0,,{{\\fad(600,0)}}{card['text']}")
    if tags:
        extra += tag_events(slots, starts, ends)
    top = os.path.join(out, "top.ass")
    open(top, "w", encoding="utf-8").write(ass.rstrip("\n") + "\n" + "\n".join(extra) + "\n")
    name = spec.get("name", "video") + ("_review" if tags else "")
    final = os.path.join(out, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", picture, "-i", song, "-filter_complex", f"[0:v]ass='{top}'[v]",
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    print(f"{made} shots in {len(slots)} slots -> {final} ({dur_of(final):.2f} s; song {total:.2f} s)")
    return final


if __name__ == "__main__":
    a = sys.argv[1:]
    tags = "--tags" in a
    a = [x for x in a if x != "--tags"]
    main(a[0], a[1], a[2], tags)
