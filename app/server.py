"""Gimme My Face: the app on your own computer.  python3 app/server.py  ->  http://localhost:8765

Standard library only. Your Comfy API key stays in this process's memory (never written to disk, never logged);
every GPU second runs on your own Comfy account. Projects live in ./projects/<name>/ (inputs) and ./out/<name>/ (outputs).
"""
import cgi
import json
import os
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(__file__))
import comfy  # noqa: E402
import make  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "app", "web")
JOBS = {}            # project -> {"log": [...], "busy": bool, "video": path|None}


def slug(s):
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")[:40] or "my-song"


def run_job(name, fn):
    job = JOBS.setdefault(name, {"log": [], "busy": False, "video": None})
    if job["busy"]:
        return False
    job.update(busy=True)

    def go():
        try:
            job["video"] = fn(lambda m: job["log"].append(m))
        except Exception as e:  # show the error on the page instead of dying quietly
            job["log"].append(f"stopped: {e}")
            traceback.print_exc()
        finally:
            job["busy"] = False
    threading.Thread(target=go, daemon=True).start()
    return True


def paths(name):
    """App projects live apart from hand-made ones: projects/web/<name> (your inputs), out/web/<name> (outputs)."""
    return os.path.join(ROOT, "projects", "web", name), os.path.join(ROOT, "out", "web", name)


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):   # quiet (and never prints form fields)
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            page = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
            page = page.replace("/*CSS*/", open(os.path.join(WEB, "mixtape.css"), encoding="utf-8").read())
            return self._send(200, page, "text/html; charset=utf-8")
        if u.path == "/api/projects":            # pick up where you left off
            od = os.path.join(ROOT, "projects", "web")
            return self._send(200, sorted(os.listdir(od)) if os.path.isdir(od) else [])
        if u.path == "/api/status":
            name = slug(q.get("p", ""))
            job = JOBS.get(name, {"log": [], "busy": False, "video": None})
            _, out = paths(name)
            slots = json.load(open(os.path.join(out, "slots.json"))) if os.path.exists(os.path.join(out, "slots.json")) else None
            tk = os.path.join(out, "takes")
            takes_list = sorted(os.path.relpath(os.path.join(tk, f), ROOT) for f in os.listdir(tk)) if os.path.isdir(tk) else []
            return self._send(200, {"log": job["log"][-40:], "busy": job["busy"], "takes": takes_list,
                                    "picked": os.path.exists(os.path.join(out, "look.png")),
                                    "video": job["video"] and os.path.relpath(job["video"], ROOT),
                                    "slots": slots, "has_key": comfy.has_key()})
        if u.path.startswith("/files/"):
            rel = os.path.normpath(u.path[len("/files/"):])
            full = os.path.join(ROOT, rel)
            if rel.startswith("..") or not rel.startswith(("out" + os.sep, "projects" + os.sep)) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            ctype = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg"}.get(os.path.splitext(full)[1], "application/octet-stream")
            # byte ranges, so the player can seek / fast-forward (29 Oct: "cant ff to the end")
            size = os.path.getsize(full)
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            a, b = 0, size - 1
            if m and (m.group(1) or m.group(2)):
                a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
                b = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else size - 1
            with open(full, "rb") as fh:
                fh.seek(a)
                data = fh.read(b - a + 1)
            self.send_response(206 if (a, b) != (0, size - 1) or m else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if m:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/key":
            key = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["key"].strip()
            if key:
                os.environ["COMFY_API_KEY"] = key
            return self._send(200, {"ok": True})
        if u.path == "/api/undo":                     # Dawn 29 Oct: "def need an undo button": step a scene back one edit
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            _, out = paths(slug(body["p"]))
            slots_path = os.path.join(out, "slots.json")
            spec = json.load(open(slots_path))
            sl = next(s for s in spec["slots"] if s["id"] == body["id"])
            hist = sl.get("edit_history") or []
            if not hist:
                return self._send(400, {"error": "nothing to undo"})
            sl["media"] = hist.pop()
            sl["edit_history"] = hist
            json.dump(spec, open(slots_path, "w"), indent=1)
            png = next((m for m in sl["media"] if m.endswith(".png")), "")
            return self._send(200, {"ok": True, "media": os.path.relpath(png, ROOT) if os.path.isabs(png) else png, "left": len(hist)})
        if u.path == "/api/erase":                    # the eraser: paint a spot, it's filled in (local, free, instant)
            import base64
            import faceswap
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = slug(body["p"])
            _, out = paths(name)
            slots_path = os.path.join(out, "slots.json")
            spec = json.load(open(slots_path))
            sl = next(s for s in spec["slots"] if s["id"] == body["id"])
            src = next(m for m in sl["media"] if m.endswith(".png"))
            src_full = src if os.path.isabs(src) else os.path.join(ROOT, src)
            os.makedirs(os.path.join(out, "erase"), exist_ok=True)
            n = len(os.listdir(os.path.join(out, "erase")))
            mask = os.path.join(out, "erase", f"{sl['id']}_mask{n}.png")
            open(mask, "wb").write(base64.b64decode(body["mask"].split(",", 1)[1]))
            dst = os.path.join(out, "erase", f"{sl['id']}_erased{n}.png")
            import subprocess
            subprocess.run([os.path.join(faceswap.FF, ".venv", "bin", "python"), "-I", os.path.join(ROOT, "app", "erase.py"),
                            src_full, mask, dst], capture_output=True)
            if not os.path.exists(dst):
                return self._send(500, {"error": "eraser failed"})
            sl.setdefault("edit_history", []).append(list(sl["media"]))   # for undo
            # the scene's motion clip was made from the picture BEFORE this edit, so it would bring the fixed spot back
            # (29 Oct: forehead marks erased on stills still showed in #04/#10/#16/#19/#22's clips): drop it; the still plays
            sl["media"] = [dst if m == src else m for m in sl["media"] if not m.endswith(".mp4")]
            json.dump(spec, open(slots_path, "w"), indent=1)
            return self._send(200, {"ok": True, "media": os.path.relpath(dst, ROOT)})
        if u.path in ("/api/start", "/api/preview"):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            name = slug(form.getfirst("name", "") or os.path.splitext(form["song"].filename)[0])
            proj, out = paths(name)
            os.makedirs(proj, exist_ok=True)
            song = os.path.join(proj, "song" + os.path.splitext(form["song"].filename)[1].lower())
            open(song, "wb").write(form["song"].file.read())
            lyr = os.path.join(proj, "lyrics.txt")
            open(lyr, "w", encoding="utf-8").write(form.getfirst("lyrics", ""))
            faces = []
            for i, f in enumerate([f for f in (form["faces"] if isinstance(form["faces"], list) else [form["faces"]])
                                   if getattr(f, "filename", None)][:3] if "faces" in form else []):
                p = os.path.join(proj, f"face{i + 1}{os.path.splitext(f.filename)[1].lower()}")
                open(p, "wb").write(f.file.read())
                faces.append(p)
            settings = {k: form.getfirst(k, "") for k in ("vibe", "story", "who", "motion", "pin", "endcard", "realface")}
            json.dump(settings, open(os.path.join(proj, "settings.json"), "w"), indent=1)
            if u.path == "/api/preview":       # free: no GPU, no key
                import lyricbeat
                import preview

                def fn(cb):
                    cb("timing your lyrics + finding the beat (on this computer, free)…")
                    lyricbeat.main(song, lyr, out)
                    preview.main(song, out)
                    cb("free preview ready: your captions + a colour block per bar")
                    return os.path.join(out, "preview.mp4")
            else:
                if not comfy.has_key():
                    return self._send(400, {"error": "Add your Comfy API key first (it pays for your own GPU time)."})
                if not faces:
                    return self._send(400, {"error": "Add 1-3 photos of your face."})
                import takes

                def fn(cb):
                    cb("making 3 takes of you…")
                    takes.make_takes(out, faces, settings["who"], progress=cb)
                    cb("IS THIS YOU? pick your take")
                    return None
            ok = run_job(name, fn)
            return self._send(200 if ok else 409, {"project": name, "started": ok})
        if u.path in ("/api/pick", "/api/reroll", "/api/recut", "/api/approve"):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = slug(body["p"])
            proj, out = paths(name)
            st = json.load(open(os.path.join(proj, "settings.json")))
            song = [os.path.join(proj, f) for f in os.listdir(proj) if f.startswith("song")][0]
            lyr = os.path.join(proj, "lyrics.txt")
            faces = sorted(os.path.join(proj, f) for f in os.listdir(proj) if f.startswith("face"))
            look = os.path.join(out, "look.png")
            slots_path = os.path.join(out, "slots.json")

            def fn(cb):
                import assemble
                import scenes
                import takes
                if u.path == "/api/pick":
                    takes.pick(out, int(body["take"]))
                    cb("locked: that face goes in every scene")
                    return make.make(song, lyr, out, faces, st["vibe"] or "dark, moody, cinematic night", st["story"],
                                     look, st["motion"] or "key", float(st["pin"] or 0.7), cb, st["endcard"] or None,
                                     st["who"], st.get("realface", "on") != "off")
                if u.path == "/api/reroll":           # "Not me" / re-roll one scene
                    spec = json.load(open(slots_path))
                    for sl in spec["slots"]:
                        if sl["id"] == body["id"]:
                            if body.get("prompt"):
                                sl["prompt"] = body["prompt"]
                            sl["media"] = []
                            sl.pop("media_orig", None)     # else the face pass re-swaps the old picture
                    json.dump(spec, open(slots_path, "w"), indent=1)
                    cb(f"re-rolling scene {body['id']}…")
                    scenes.main(slots_path, out, faces, [body["id"]], bool(body.get("motion")),
                                float(body.get("pin") or st["pin"] or 0.7), look if os.path.exists(look) else None)
                    import faceswap
                    if st.get("realface", "on") != "off" and faces and faceswap.available():
                        cb(f"real face on scene {body['id']}…")
                        faceswap.main(slots_path, faces[0], cb)
                    if not json.load(open(slots_path)).get("approved"):   # still in the scene check: back to it
                        cb("CHECK: is every scene you? tap ✓ me or ↻ not me on each one")
                        return None
                if u.path == "/api/approve":          # every scene OK'd: cut the video
                    spec = json.load(open(slots_path))
                    spec["approved"] = True
                    json.dump(spec, open(slots_path, "w"), indent=1)
                    cb("4/4 cutting the video on the beat + captions on top…")
                    return assemble.main(song, out, slots_path)
                cb("re-cutting the video…")
                return assemble.main(song, out, slots_path)
            ok = run_job(name, fn)
            return self._send(200 if ok else 409, {"started": ok})
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Gimme My Face -> http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
