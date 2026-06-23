"""
Relight Studio backend.

Serves the frontend studio and exposes AI relighting endpoints:
  GET  /                 -> the studio UI (index.html)
  GET  /api/status       -> engine / GPU status
  POST /api/warmup       -> begin loading models (non-blocking)
  POST /api/relight      -> LOCAL IC-Light relight (multipart image + params)
  POST /api/relight_cloud-> ONLINE relight via fal.ai (needs FAL_KEY) — highest
                            quality fallback for machines without a capable GPU

Run:  python backend/server.py     (or use start.bat)
"""
import io
import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(title="Relight Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_infer_lock = threading.Lock()


# ---- lazy import so the server starts instantly (heavy torch import deferred) ----
def _engine():
    import ic_light_engine as e
    return e


@app.get("/api/status")
def api_status():
    try:
        return _engine().status()
    except Exception as e:
        return {"loaded": False, "error": repr(e), "progress": "import-failed"}


@app.post("/api/warmup")
def api_warmup():
    eng = _engine()

    def _bg():
        try:
            eng.load()
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_bg, daemon=True).start()
    return eng.status()


def _depth():
    import depth_engine as d
    return d


def _seg():
    import seg_engine as s
    return s


@app.post("/api/segment")
def api_segment(image: UploadFile = File(...)):
    """Detect the object and remove the background. Returns an RGBA cutout PNG."""
    try:
        pil = Image.open(io.BytesIO(image.file.read()))
    except Exception:
        raise HTTPException(400, "Could not read uploaded image")
    with _infer_lock:
        try:
            cut = _seg().segment(pil)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Segmentation failed: {e!r}")
    return _img_to_png_response(cut)


@app.get("/api/seg_status")
def api_seg_status():
    try:
        return _seg().status()
    except Exception as e:
        return {"loaded": False, "error": repr(e)}


@app.post("/api/depth")
def api_depth(image: UploadFile = File(...)):
    """Estimate a depth map (MiDaS) used by the 3D studio for real geometry."""
    try:
        pil = Image.open(io.BytesIO(image.file.read()))
    except Exception:
        raise HTTPException(400, "Could not read uploaded image")
    with _infer_lock:
        try:
            depth = _depth().estimate(pil)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Depth estimation failed: {e!r}")
    return _img_to_png_response(depth)


@app.get("/api/depth_status")
def api_depth_status():
    try:
        return _depth().status()
    except Exception as e:
        return {"loaded": False, "error": repr(e)}


def _img_to_png_response(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/relight")
def api_relight(
    image: UploadFile = File(...),
    prompt: str = Form("cinematic lighting"),
    bg_source: str = Form("None"),
    width: int = Form(512),
    height: int = Form(640),
    steps: int = Form(25),
    cfg: float = Form(2.0),
    seed: int = Form(12345),
    highres_scale: float = Form(1.0),
    highres_denoise: float = Form(0.5),
    a_prompt: str = Form("best quality"),
    n_prompt: str = Form("lowres, bad anatomy, bad hands, cropped, worst quality"),
):
    try:
        pil = Image.open(io.BytesIO(image.file.read()))
    except Exception:
        raise HTTPException(400, "Could not read uploaded image")
    eng = _engine()
    with _infer_lock:
        try:
            result, _fg = eng.relight(
                pil, prompt=prompt, bg_source=bg_source, width=width, height=height,
                steps=steps, cfg=cfg, seed=seed, highres_scale=highres_scale,
                highres_denoise=highres_denoise, a_prompt=a_prompt, n_prompt=n_prompt)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Relight failed: {e!r}")
    return _img_to_png_response(result)


@app.post("/api/relight_cloud")
def api_relight_cloud(
    image: UploadFile = File(...),
    prompt: str = Form("cinematic lighting"),
    bg_source: str = Form("None"),
):
    """Highest-quality online path via fal.ai IC-Light v2. Requires FAL_KEY env var."""
    key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not key:
        raise HTTPException(400, "Online mode needs a FAL_KEY environment variable (get one free at fal.ai).")
    import base64, requests
    raw = image.file.read()
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    direction = {"None": "None", "Left Light": "Left", "Right Light": "Right",
                 "Top Light": "Top", "Bottom Light": "Bottom"}.get(bg_source, "None")
    payload = {"image_url": data_url, "prompt": prompt, "initial_latent": direction}
    try:
        r = requests.post("https://fal.run/fal-ai/iclight-v2",
                          headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                          json=payload, timeout=180)
        r.raise_for_status()
        out = r.json()
        url = out["images"][0]["url"] if "images" in out else out["image"]["url"]
        img_bytes = requests.get(url, timeout=120).content if url.startswith("http") \
            else base64.b64decode(url.split(",", 1)[1])
        return _img_to_png_response(Image.open(io.BytesIO(img_bytes)))
    except requests.HTTPError as e:
        raise HTTPException(502, f"fal.ai error: {e} -> {r.text[:300]}")
    except Exception as e:
        raise HTTPException(500, f"Cloud relight failed: {e!r}")


# ---- serve the frontend (mounted last so /api/* wins) ----
@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "index.html"))


app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"\n  Relight Studio running:  http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
