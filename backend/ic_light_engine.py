"""
IC-Light relighting engine (foreground-conditioned model).

Faithful adaptation of lllyasviel/IC-Light's gradio_demo.py into a reusable,
lazy-loading module with low-VRAM optimizations suitable for an 8GB GPU.

Produces photorealistic relighting from a single photo + a text prompt + a
light-direction preference, while preserving the subject (composition / pose).
"""
import os
import math
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from enum import Enum

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
# Keep HF cache local to the project so everything stays self-contained.
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(MODELS_DIR, "hf")))

SD15_NAME = "stablediffusionapi/realistic-vision-v51"
IC_LIGHT_URL = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"
IC_LIGHT_PATH = os.path.join(MODELS_DIR, "iclight_sd15_fc.safetensors")


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


# ----- module-level singletons (filled by load()) -----
_state = {"loaded": False, "loading": False, "error": None, "progress": "idle"}
device = None
tokenizer = text_encoder = vae = unet = rmbg = None
t2i_pipe = i2i_pipe = None
unet_original_forward = None


def status():
    s = dict(_state)
    s["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        s["gpu"] = torch.cuda.get_device_name(0)
    return s


def _set(p):
    _state["progress"] = p
    print(f"[engine] {p}", flush=True)


def load():
    """Lazy-load all models. Safe to call repeatedly."""
    global device, tokenizer, text_encoder, vae, unet, rmbg
    global t2i_pipe, i2i_pipe, unet_original_forward
    if _state["loaded"]:
        return
    if _state["loading"]:
        return
    _state["loading"] = True
    _state["error"] = None
    try:
        from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                               AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler)
        from diffusers.models.attention_processor import AttnProcessor2_0
        from transformers import CLIPTextModel, CLIPTokenizer
        from torch.hub import download_url_to_file
        from briarmbg import BriaRMBG

        _set("downloading / loading SD1.5 (realistic-vision-v51)…")
        tokenizer = CLIPTokenizer.from_pretrained(SD15_NAME, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(SD15_NAME, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(SD15_NAME, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(SD15_NAME, subfolder="unet")

        _set("downloading / loading background-removal model (RMBG-1.4)…")
        rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        # expand UNet input conv 4 -> 8 channels (noisy latent + fg condition)
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels,
                                          unet.conv_in.kernel_size, unet.conv_in.stride,
                                          unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias = unet.conv_in.bias
            unet.conv_in = new_conv_in

        unet_original_forward = unet.forward

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

        unet.forward = hooked_unet_forward

        _set("downloading IC-Light weights…")
        if not os.path.exists(IC_LIGHT_PATH):
            download_url_to_file(url=IC_LIGHT_URL, dst=IC_LIGHT_PATH)

        _set("merging IC-Light weights into UNet…")
        sd_offset = sf.load_file(IC_LIGHT_PATH)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            text_encoder = text_encoder.to(device=device, dtype=torch.float16)
            vae = vae.to(device=device, dtype=torch.bfloat16)
            unet = unet.to(device=device, dtype=torch.float16)
            rmbg = rmbg.to(device=device, dtype=torch.float32)
        else:
            text_encoder = text_encoder.to(device)
            vae = vae.to(device)
            unet = unet.to(device)
            rmbg = rmbg.to(device)

        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())
        # low-VRAM: tile + slice VAE so 8GB can handle highres decode
        try:
            vae.enable_tiling()
            vae.enable_slicing()
        except Exception:
            pass

        sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1)

        common = dict(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                      scheduler=sched, safety_checker=None, requires_safety_checker=False,
                      feature_extractor=None, image_encoder=None)
        t2i_pipe = StableDiffusionPipeline(**common)
        i2i_pipe = StableDiffusionImg2ImgPipeline(**common)

        _state["loaded"] = True
        _set("ready")
    except Exception as e:
        _state["error"] = repr(e)
        _set(f"ERROR: {e!r}")
        raise
    finally:
        _state["loading"] = False


# ---------------- helpers (from reference) ----------------
@torch.inference_mode()
def _encode_prompt_inner(txt):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start, id_end = tokenizer.bos_token_id, tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i:i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]
    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    return text_encoder(token_ids).last_hidden_state


@torch.inference_mode()
def _encode_prompt_pair(positive, negative):
    c = _encode_prompt_inner(positive)
    uc = _encode_prompt_inner(negative)
    c_len, uc_len = float(len(c)), float(len(uc))
    c_repeat = int(math.ceil(max(c_len, uc_len) / c_len))
    uc_repeat = int(math.ceil(max(c_len, uc_len) / uc_len))
    max_chunk = max(len(c), len(uc))
    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]
    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)
    return c, uc


@torch.inference_mode()
def _pt2np(imgs):
    out = []
    for x in imgs:
        y = x.movedim(0, -1) * 127.5 + 127.5
        out.append(y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
    return out


@torch.inference_mode()
def _np2pt(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    return h.movedim(-1, 1)


def _resize_center_crop(image, tw, th):
    pil = Image.fromarray(image)
    ow, oh = pil.size
    s = max(tw / ow, th / oh)
    rw, rh = int(round(ow * s)), int(round(oh * s))
    pil = pil.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - tw) / 2, (rh - th) / 2
    return np.array(pil.crop((left, top, left + tw, top + th)))


def _resize(image, tw, th):
    return np.array(Image.fromarray(image).resize((tw, th), Image.LANCZOS))


@torch.inference_mode()
def _run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    k = (256.0 / float(H * W)) ** 0.5
    feed = _resize(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = _np2pt([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0].detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


def _bg_latent_init(bg_source, w, h):
    if bg_source == BGSource.NONE:
        return None
    if bg_source == BGSource.LEFT:
        grad = np.linspace(255, 0, w); img = np.tile(grad, (h, 1))
    elif bg_source == BGSource.RIGHT:
        grad = np.linspace(0, 255, w); img = np.tile(grad, (h, 1))
    elif bg_source == BGSource.TOP:
        grad = np.linspace(255, 0, h)[:, None]; img = np.tile(grad, (1, w))
    elif bg_source == BGSource.BOTTOM:
        grad = np.linspace(0, 255, h)[:, None]; img = np.tile(grad, (1, w))
    else:
        raise ValueError("bad bg_source")
    return np.stack((img,) * 3, axis=-1).astype(np.uint8)


@torch.inference_mode()
def _process(input_fg, prompt, w, h, num_samples, seed, steps, a_prompt, n_prompt,
             cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_bg = _bg_latent_init(bg_source, w, h)
    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = _resize_center_crop(input_fg, w, h)
    concat_conds = _np2pt([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = _encode_prompt_pair(prompt + ", " + a_prompt, n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds, negative_prompt_embeds=unconds, width=w, height=h,
            num_inference_steps=steps, num_images_per_prompt=num_samples, generator=rng,
            output_type="latent", guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = _resize_center_crop(input_bg, w, h)
        bg_latent = _np2pt([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent, strength=lowres_denoise, prompt_embeds=conds,
            negative_prompt_embeds=unconds, width=w, height=h,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples, generator=rng, output_type="latent",
            guidance_scale=cfg, cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = _pt2np(pixels)
    pixels = [_resize(p, int(round(w * highres_scale / 64.0) * 64),
                      int(round(h * highres_scale / 64.0) * 64)) for p in pixels]

    pixels = _np2pt(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)
    h2, w2 = latents.shape[2] * 8, latents.shape[3] * 8

    fg = _resize_center_crop(input_fg, w2, h2)
    concat_conds = _np2pt([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents, strength=highres_denoise, prompt_embeds=conds,
        negative_prompt_embeds=unconds, width=w2, height=h2,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples, generator=rng, output_type="latent",
        guidance_scale=cfg, cross_attention_kwargs={"concat_conds": concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    return _pt2np(pixels)


@torch.inference_mode()
def relight(image: Image.Image, prompt: str, bg_source: str = "None",
            width: int = 512, height: int = 640, steps: int = 25, cfg: float = 2.0,
            seed: int = 12345, highres_scale: float = 1.5, highres_denoise: float = 0.5,
            lowres_denoise: float = 0.9,
            a_prompt: str = "best quality",
            n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality"):
    """Relight a PIL image. Returns (result PIL image, preprocessed-foreground PIL image)."""
    load()
    rgb = np.array(image.convert("RGB"))
    fg, _ = _run_rmbg(rgb)
    results = _process(fg, prompt, int(width), int(height), 1, seed, steps,
                       a_prompt, n_prompt, cfg, highres_scale, highres_denoise,
                       lowres_denoise, BGSource(bg_source))
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    return Image.fromarray(results[0]), Image.fromarray(fg)
