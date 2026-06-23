"""
Monocular depth estimation (MiDaS) used to turn a flat photo into 3D geometry.

The frontend uses the returned depth map as a displacement map so the photo's
subject becomes a real 3D surface that casts and receives shadows — while the
original photo pixels stay untouched as the albedo (composition preserved).
"""
import os
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TORCH_HOME", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "torch")))

_state = {"loaded": False, "loading": False, "error": None, "model": "MiDaS DPT_Hybrid"}
_midas = None
_transform = None
_device = None


def status():
    s = dict(_state)
    s["cuda"] = torch.cuda.is_available()
    return s


def load():
    global _midas, _transform, _device
    if _state["loaded"] or _state["loading"]:
        return
    _state["loading"] = True
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DPT_Hybrid = good accuracy/speed balance; needs `timm`.
        _midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        _midas.to(_device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _transform = transforms.dpt_transform
        _state["loaded"] = True
        print("[depth] MiDaS ready", flush=True)
    except Exception as e:
        _state["error"] = repr(e)
        print(f"[depth] ERROR {e!r}", flush=True)
        raise
    finally:
        _state["loading"] = False


@torch.inference_mode()
def estimate(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """Return an 8-bit grayscale depth map (brighter = closer)."""
    load()
    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    batch = _transform(rgb).to(_device)
    pred = _midas(batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False).squeeze()
    d = pred.detach().float().cpu().numpy()
    # MiDaS outputs inverse depth (larger = closer). Normalize to 0..255.
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    depth_img = Image.fromarray((d * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    # cap output size for the browser displacement mesh
    if max(W, H) > max_side:
        s = max_side / max(W, H)
        depth_img = depth_img.resize((int(W * s), int(H * s)), Image.BILINEAR)
    if _device is not None and _device.type == "cuda":
        torch.cuda.empty_cache()
    return depth_img
