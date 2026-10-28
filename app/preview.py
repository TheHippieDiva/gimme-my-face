"""Preview: the song + karaoke captions + a colour cut on every bar, as a vertical video.

    python3 app/preview.py SONG.mp3 OUT_DIR

Reads OUT_DIR/timeline.json + OUT_DIR/captions.ass from lyricbeat.py and writes OUT_DIR/preview.mp4.
The colour blocks stand in for the scenes Comfy will make (one scene per bar); a thin flash marks
each beat, so you can see the cuts land on the music before any GPU time is spent.
"""
import json, subprocess, sys

PALETTE = ["0x2b1d3a", "0x6b3fa0", "0xd49066", "0x1f4e5f", "0x8e2c48", "0x3a5a40", "0xc9a227", "0x264653"]


def main(song, out):
    tl = json.load(open(f"{out}/timeline.json"))
    beats = tl["beats"]
    period = 60.0 / tl["bpm"]
    b0 = beats[0]
    bars = beats[::4]                       # one scene per bar of 4 beats
    dur = float(subprocess.run(["ffmpeg", "-hide_banner", "-i", song], capture_output=True, text=True)
                .stderr.split("Duration: ")[1].split(",")[0].split(":")[-1]) + 60 * int(
        subprocess.run(["ffmpeg", "-hide_banner", "-i", song], capture_output=True, text=True)
        .stderr.split("Duration: ")[1].split(",")[0].split(":")[1])
    # background: one solid colour per bar, joined with the concat filter
    edges = [0.0] + bars[1:] + [dur]
    segs, f = [], []
    for i in range(len(edges) - 1):
        d = max(0.04, edges[i + 1] - edges[i])
        f.append(f"color=c={PALETTE[i % len(PALETTE)]}:s=1080x1920:r=30:d={d:.3f}[c{i}]")
        segs.append(f"[c{i}]")
    f.append("".join(segs) + f"concat=n={len(segs)}:v=1:a=0[bg]")
    flash = f"lt(mod(t-{b0:.3f}\\,{period:.4f})\\,0.07)*gte(t\\,{b0:.3f})"
    f.append(f"[bg]drawbox=x=0:y=1880:w=1080:h=40:color=white@0.8:t=fill:enable='{flash}',"
             f"ass='{out}/captions.ass'[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", song, "-filter_complex", ";".join(f), "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", f"{out}/preview.mp4"], check=True)
    print("preview ->", f"{out}/preview.mp4", f"({len(bars)} scenes, {tl['bpm']} bpm)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
