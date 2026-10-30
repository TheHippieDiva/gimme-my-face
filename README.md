# Gimme My Face · Open Source Higgsfield entry

**My face. My song. My video.** Your song + your lyrics + a few photos of your face → a vertical music video where the
face in every scene stays *you*: cuts on the beat, karaoke captions on top. One button on top, a pro drawer underneath.
Built on the Comfy Cloud API by Dawn de Brantes (TheHippieDiva) with Claude.

## Run it
```
pip install faster-whisper numpy pillow     # + ffmpeg (with libass) on your PATH; macOS for the on-device text check
python3 app/server.py                        # open http://localhost:8765
```
Paste your own Comfy API key (platform.comfy.org → API Keys; any paid Comfy plan). It stays in memory on your computer
and every GPU second bills your own Comfy account. No key? **Free preview** times your lyrics to the beat on your laptop.

Command line, same thing:
```
python3 app/make.py SONG.mp3 LYRICS.txt out/mysong FACE1.jpg FACE2.jpg --vibe "neon 80s night" --motion key
```

## How it works
| Step | File | Where it runs |
|---|---|---|
| 1. Words timed to the song (your lyrics are the truth, the speech model is only the clock) + beat tracker | `app/lyricbeat.py` | your computer, free |
| 2. Storyboard: each line becomes a picture idea; never quotes lyrics, so no words land in a picture | `app/storyboard.py` | Claude node on Comfy |
| 3. A scene with your face: Qwen Image Edit builds the scene from your photos, a 2nd edit pass locks the face to your look; on-device OCR rejects any picture with letters | `app/scenes.py`, `app/qwen_face.py`, `tools/textcheck.js` | Comfy Cloud |
| 4. **Face pin** motion: LTX-2 image→video with your still on the first frame *and* a tighter crop of it guided onto the last frame, so the camera pushes in and the face can't drift into a stranger | `app/ltx2.py` | Comfy Cloud |
| 4b. **Real face**: your face from one clear front photo goes onto every finished scene with FaceFusion (hyperswap_1a_256, no enhancer). Generated faces drift; this pins them to you. Switch off in the pro drawer ("AI face only") | `app/faceswap.py` | your computer, free |
| 5. Assembly: a cut every bar or two, captions + end card as the only top layer, 1080×1920 | `app/assemble.py` | your computer |

Pro drawer: re-roll any scene (edit its picture idea), motion off / key moments / every scene, face strength (pin),
ink description, end card, re-cut. Every scene is saved clean (no captions) and can be swapped for real footage in
`out/<song>/slots.json` (`media`).

### Made for people who aren't AI people
- **Build your character** (like an RPG): I am, age, build, hair, hair colour, extras, story genre, music style. It becomes
  one sentence pinned into every scene, and the storyboard picks **one outfit for the whole video**, so the person stays the
  same person from scene to scene.
- **3 photo slots that say what to shoot**: straight on, three-quarter, big smile.
- **"Is this you?" twice**: pick your face from 3 takes, then check **every scene** before the cut. On each scene:
  **✓ me · ⌫ erase · ✎ fix · ↻ not me**.
  - **⌫ Magic eraser + 🎨 colour brush** (`app/erase.py`) run on your computer, free and instant: paint out a spot, or tap a
    colour and paint it where it belongs. Every fix you make by hand is a GPU re-roll you don't pay for.
  - **✎ fix** re-rolls one scene with targeted fixes (weird mark, hair, grey hair, glasses, eyes, face shape, ink, outfit).
- **Always shows it's working**: a live "cooking… simmering…" status, % done, minutes left, a "come back around 9:41"
  time and a ding when it's ready.
- Pink always means GO.

### Workflow files
The app builds its Comfy graphs in Python so it can fill in your photos and prompts. The same graphs, in ComfyUI API
format with placeholder inputs, are in [`workflows/`](workflows/) (re-export with `python3 tools/export_workflows.py`).

## Real face (optional, recommended)
Install [FaceFusion](https://github.com/facefusion/facefusion) in `~/tools/facefusion` (or set `FACEFUSION_DIR`). Without it the app still runs and skips this step. Note: the face-swap models are licensed for non-commercial use; a paid product needs a commercial license from their makers.

## Made with
- **Comfy Cloud API** runs the GPU work: Qwen Image Edit (scenes with your face), LTX-2 (face-pinned motion), Wan 2.2 Animate (your moves), and Claude on Comfy (the storyboard).
- **FaceFusion** (optional, on your computer) puts your real face on the finished scenes.
- **faster-whisper** times your lyrics to the song; **ffmpeg + libass** cut the video and burn the karaoke.
- The demo video "AI N00b" also uses stills from **Grok Imagine** (outside the app), finished with the app's real-face step.
- Built by Dawn de Brantes (TheHippieDiva) with Claude.

## Your face, your rules
Only use faces you have the right to use: your own, or people who said yes. The app never uploads your photos anywhere
except your own Comfy account for the GPU steps; the real-face step runs on your computer.

## License
MIT (see LICENSE).
