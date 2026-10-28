"""Tiny Comfy Cloud client (standard library only).

Key: COMFY_API_KEY in .env (never printed, never committed).
Docs: https://docs.comfy.org/development/cloud/api-reference
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://cloud.comfy.org"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _key():
    if os.environ.get("COMFY_API_KEY"):            # bring-your-own-key: the app sets this per user
        return os.environ["COMFY_API_KEY"].strip()
    env = os.path.join(ROOT, ".env")
    for line in (open(env) if os.path.exists(env) else []):
        if line.startswith("COMFY_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("No Comfy API key: paste it in the app, or put COMFY_API_KEY=... in .env")


def has_key():
    try:
        return bool(_key())
    except SystemExit:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _req(method, path, body=None, params=None, follow=True):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"X-API-Key": _key(), "Content-Type": "application/json"})
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(r, timeout=60)
    except urllib.error.HTTPError as e:
        if not follow and e.code in (301, 302, 303, 307):
            return e
        raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}")


def upload_image(path):
    """Upload a local image as a Comfy input; returns the name to use in LoadImage."""
    import uuid
    b = uuid.uuid4().hex
    name = os.path.basename(path)
    body = b"".join([
        f"--{b}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode(), open(path, "rb").read(), b"\r\n",
        f"--{b}--\r\n".encode()])
    r = urllib.request.Request(BASE + "/api/upload/image", data=body, method="POST",
                               headers={"X-API-Key": _key(), "Content-Type": f"multipart/form-data; boundary={b}"})
    j = json.load(urllib.request.urlopen(r, timeout=120))
    return (j.get("subfolder", "") + "/" if j.get("subfolder") else "") + j["name"]


def submit(workflow):
    # extra_data key = partner nodes (e.g. ClaudeNode) bill the same Comfy account
    body = {"prompt": workflow, "extra_data": {"api_key_comfy_org": _key()}}
    return json.load(_req("POST", "/api/prompt", body))["prompt_id"]


def job(prompt_id):
    return json.load(_req("GET", f"/api/jobs/{prompt_id}"))


def wait(prompt_id, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        s = json.load(_req("GET", f"/api/job/{prompt_id}/status"))["status"]
        if s == "success":
            return
        if s in ("error", "non_retryable_error", "lost", "cancelled"):
            raise RuntimeError(f"job {prompt_id}: {s}")
        time.sleep(3)
    raise TimeoutError(prompt_id)


def outputs(prompt_id):
    """File entries ({filename, subfolder, type}) from the finished job."""
    h = json.load(_req("GET", f"/api/jobs/{prompt_id}"))  # Comfy Cloud: /api/history is not available
    files = []
    for node in (h.get("outputs") or {}).values():
        for kind in ("images", "video", "videos", "gifs", "audio"):
            files += node.get(kind, [])
    return files


def download(f, dest):
    r = _req("GET", "/api/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                         "type": f.get("type", "output")}, follow=False)
    loc = r.headers.get("Location")
    data = urllib.request.urlopen(loc, timeout=120).read() if loc else r.read()  # signed URL: no key
    with open(dest, "wb") as fh:
        fh.write(data)
    return dest
