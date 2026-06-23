"""
Object detection + background removal (RMBG-1.4), standalone.

Returns an RGBA cutout (object isolated, background transparent). The alpha
channel doubles as the object mask used for multi-view silhouette alignment.
Loads only the small RMBG model (not the full IC-Light stack).
"""
import os
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))

_state = {"loaded": False, "loading": False, "error": None, "model": "RMBG-1.4"}
_rmbg = None
_device = None


def status():
    s = dict(_state)
    s["cuda"] = torch.cuda.is_available()
    return s


def load():
    global _rmbg, _device
    if _state["loaded"] or _state["loading"]:
        return
    _state["loading"] = True
    try:
        from briarmbg import BriaRMBG
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        _rmbg = _rmbg.to(device=_device, dtype=torch.float32).eval()
        _state["loaded"] = True
        print("[seg] RMBG ready", flush=True)
    except Exception as e:
        _state["error"] = repr(e)
        print(f"[seg] ERROR {e!r}", flush=True)
        raise
    finally:
        _state["loading"] = False


def _resize(img, w, h):
    return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))


@torch.inference_mode()
def segment(image: Image.Image):
    """Return an RGBA PIL image (object kept, background transparent via alpha=mask).

    Preprocessing mirrors IC-Light's run_rmbg (the known-good path): feed ~256px²,
    normalize with /127 - 1, sigmoid mask output in [0,1].
    """
    load()
    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    k = (256.0 / float(H * W)) ** 0.5
    fw, fh = max(64, int(64 * round(W * k))), max(64, int(64 * round(H * k)))
    feed = _resize(rgb, fw, fh)
    t = torch.from_numpy(feed[None].astype(np.float32)) / 127.0 - 1.0
    t = t.movedim(-1, 1).to(_device)
    alpha = _rmbg(t)[0][0]  # [1,1,h,w]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
    a = alpha.movedim(1, -1)[0].squeeze().detach().float().cpu().numpy().clip(0, 1)
    alpha_img = (a * 255.0).astype(np.uint8)
    out = np.dstack([rgb, alpha_img]).astype(np.uint8)
    if _device is not None and _device.type == "cuda":
        torch.cuda.empty_cache()
    return Image.fromarray(out, mode="RGBA")
