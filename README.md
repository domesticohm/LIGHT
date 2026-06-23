# ◐ Relight Studio

A local **virtual photo-lighting studio** with two engines:

1. **Instant WebGL studio** — real-time, GPU-accelerated relighting in the browser.
   360° light positioning, intensity, softness/hardness, diffusion, shadow
   strength/falloff, color temperature, multiple independent lights. Zero install.
2. **🤖 AI Relight (photorealistic)** — powered by **[IC-Light](https://github.com/lllyasviel/IC-Light)**
   (by the author of ControlNet), the state of the art for single-image relighting.
   Produces hyperrealistic, physically-consistent lighting and shadows from a text
   prompt + light direction, while preserving your subject. Runs **locally on your GPU**,
   or **online via fal.ai** for the highest quality on machines without a strong GPU.

3. **🧊 3D Studio — real cast shadows (Unreal-style control)** — the photo is turned
   into a real 3D surface via depth estimation (MiDaS, on your GPU) and lit with a true
   shadow-mapping light rig in **Three.js**. The **original photo pixels are never
   altered** — only the lighting changes, so composition and design are preserved exactly.
   Controls: light direction, shadow strength, soft-light (shadow softness), night/day,
   ambient, color temperature, focal length, a **Film Noir** look, and you can **import a
   3D model** (.glb/.gltf/.obj) that casts/receives shadows with the photo.

Everything can run **100% offline** (after the one-time model download).

---

## Quick start

### Option A — full app (WebGL + AI), recommended
1. Double-click **`Relight Studio.bat`**.
2. It launches the local backend and opens **http://localhost:7860**.
3. Open an image, tweak lights live in the right panel, then hit
   **✨ Render Photorealistic Relight** for the AI pass.

> The **first** AI render downloads ~5 GB of models (SD1.5 realistic-vision +
> IC-Light + RMBG background remover) into `./models/`. After that it's offline
> and renders in seconds on a modern GPU.

### Option B — instant WebGL only (no install, no server)
Just open **`index.html`** in your browser. The AI button needs the backend, but
all real-time lighting controls work standalone.

---

## Controls

**WebGL studio (live):** per-light compass (360° aim), azimuth/elevation, intensity,
softness, hardness, diffusion, color temperature + tint; global ambient, exposure,
surface relief, shadow strength/falloff, preserve-original, specular; finishing
(contrast, saturation, vignette, bloom). Hold **Compare** for the original; **Save PNG** exports.

**AI Relight:** lighting prompt, dramatic presets (Cinematic, Golden hour, Studio,
Sunset rim, Neon noir, Window light, Hard noir, Moonlight), initial light direction,
quality steps, light influence (CFG), detail-preserve, output size, and Local/Online engine.

---

## Online (highest-quality) mode
Online mode uses fal.ai's hosted IC-Light v2. Set a key before launching:
```
setx FAL_KEY "your-fal-key"
```
Get a key at https://fal.ai. Without a key, the app uses the **local** engine.

---

## Requirements
- Windows, NVIDIA GPU recommended (tested on RTX 4070 Laptop, 8 GB). CPU works but is slow.
- Python 3.11 (an isolated `.venv` is created by the setup; PyTorch has no 3.14 wheels).
- ~6 GB free disk for models.

## Tech
- Frontend: single-file WebGL fragment-shader relighter (`index.html`).
- Backend: FastAPI (`backend/server.py`) + IC-Light engine (`backend/ic_light_engine.py`).
- Models auto-downloaded from Hugging Face on first run into `./models/`.

## Credits
- [IC-Light](https://github.com/lllyasviel/IC-Light) by lllyasviel (Apache-2.0).
- `briarmbg.py` (RMBG-1.4 architecture) vendored from the IC-Light repo.
