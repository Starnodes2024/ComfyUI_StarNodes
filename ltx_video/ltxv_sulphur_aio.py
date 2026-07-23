"""
LTXV Sulphur All-in-One engine node (v2).

Single-node port of the "LTXV Sulphur All in 1" workflow:
  pass 1 at half resolution -> 2x latent upscale -> pass 2 at full resolution,
with selectable mode (text-to-video / image-to-video / image+audio-to-video),
baked sigma presets (8 / 12 / 16 steps), HD/FHD size presets with
ratio-from-image (Star LTX Video Settings logic), base model + up to three
LoRAs, VAE / audio-VAE / CLIP / latent-upscaler dropdowns, internal prompt
encoding - and internal caches so big files are only re-loaded when the
selection actually changes.

Nothing else is required on the canvas except LoadImage / LoadAudio for the
i2v modes and your save / upscale nodes downstream.
"""

import re
import time

import numpy as np
import torch
from PIL import Image

import folder_paths
import nodes
import comfy.sd
import comfy.samplers
import comfy.utils
import comfy.model_management  # noqa: F401  (kept for parity with loader nodes)

from ..star_progress import make_event_cb, patch_model_for_progress

from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVImgToVideoInplace,
    LTXVConditioning,
    LTXVCropGuides,
    LTXVPreprocess,
    LTXVConcatAVLatent,
    LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_audio import (
    LTXVEmptyLatentAudio,
    LTXVAudioVAEEncode,
    LTXVAudioVAEDecode,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_custom_sampler import (
    RandomNoise,
    KSamplerSelect,
    CFGGuider,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_audio import TrimAudioDuration


# ---------------------------------------------------------------------------
# constants (carried over from the original workflow, not exposed as widgets)
# ---------------------------------------------------------------------------

GUIDE_RESIZE_LONG_EDGE = 1536  # ResizeImagesByLongerEdge value in the workflow
IMG_COMPRESSION = 18           # LTXVPreprocess value in the i2v branch of the workflow
DEFAULT_NEGATIVE = "console game, video game, cartoon, childish, ugly"

# Sigma schedules taken verbatim from the workflow's note node.
SIGMA_PRESETS = {
    "8 steps": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    "12 steps": "1.0, 0.995833, 0.991667, 0.9875, 0.983333, 0.979167, 0.975, 0.93125, 0.847917, 0.725, 0.522917, 0.28125, 0.0",
    "16 steps": "1.0, 0.996875, 0.99375, 0.990625, 0.9875, 0.984375, 0.98125, 0.978125, 0.975, 0.942187, 0.909375, 0.817187, 0.725, 0.573437, 0.421875, 0.210937, 0.0",
}
DEFAULT_PASS2_SIGMAS = "0.85, 0.725, 0.6, 0.4219, 0.0"

# Size presets, copied 1:1 from the Star LTX Video Settings node.
HD_RATIOS = {
    "1:1": (1280, 1280), "4:3": (1280, 960), "3:2": (1280, 853),
    "16:10": (1280, 800), "16:9": (1280, 720), "21:9": (1280, 548),
    "3:4": (960, 1280), "2:3": (853, 1280), "10:16": (800, 1280),
    "9:16": (720, 1280), "9:21": (548, 1280),
}
FHD_RATIOS = {
    "1:1": (1920, 1920), "4:3": (1920, 1440), "3:2": (1920, 1280),
    "16:10": (1920, 1200), "16:9": (1920, 1080), "21:9": (1920, 823),
    "3:4": (1440, 1920), "2:3": (1280, 1920), "10:16": (1200, 1920),
    "9:16": (1080, 1920), "9:21": (823, 1920),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _res(out):
    """Unwrap a node result: io.NodeOutput (V3) or plain tuple (V1)."""
    r = getattr(out, "result", out)
    if r is None:
        return ()
    if isinstance(r, (list, tuple)):
        return tuple(r)
    return (r,)


def _parse_sigmas(text):
    """Same parsing as the core ManualSigmas node."""
    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.*\d+)", text)]
    if len(values) < 2:
        raise ValueError("[LTXV Sulphur AIO] sigma schedule needs at least two values")
    return torch.FloatTensor(values)


def _resize_longer_edge(image, longer_edge):
    """Same behaviour as ResizeImagesByLongerEdge (aspect preserved, LANCZOS)."""
    resized = []
    for img_t in image:
        arr = (img_t.cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
        w, h = pil.size
        if w > h:
            new_w, new_h = longer_edge, int(h * (longer_edge / w))
        else:
            new_h, new_w = longer_edge, int(w * (longer_edge / h))
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255.0))
    return torch.stack(resized)


# --- Star LTX Video Settings logic ------------------------------------------

def _round_32_plus_1(value):
    """Round to nearest (multiple of 32) + 1, as the Star settings node does."""
    return max(32, int(round((value - 1) / 32) * 32)) + 1


def _round_8_plus_1(value):
    """Round to nearest (multiple of 8) + 1, as the Star settings node does."""
    return max(8, int(round((value - 1) / 8) * 8)) + 1


def _image_ratio(image):
    if image is None:
        return None
    try:
        h, w = int(image.shape[1]), int(image.shape[2])
        return w / h if h > 0 and w > 0 else None
    except Exception:
        return None


def _resolve_size(video_size, ratio, ratio_from_image, custom_width, custom_height, image):
    """Width/height exactly like the Star LTX Video Settings node."""
    ratio_dict = HD_RATIOS if video_size == "HD" else FHD_RATIOS if video_size == "FHD" else None
    source = "preset" if ratio_dict is not None else "custom"

    img_r = _image_ratio(image) if ratio_from_image else None
    if img_r is not None and ratio_dict is not None:
        ratio = min(ratio_dict, key=lambda k: abs(ratio_dict[k][0] / ratio_dict[k][1] - img_r))
        width, height = ratio_dict[ratio]
        source = f"from image -> {ratio}"
    elif img_r is not None and video_size == "Custom":
        width, height = custom_width, int(custom_width / img_r)
        source = "from image (custom)"
    elif ratio_dict is not None:
        width, height = ratio_dict[ratio]
    else:
        width, height = custom_width, custom_height

    return _round_32_plus_1(width), _round_32_plus_1(height), source


def _encode_prompt(clip, text):
    """Same as CLIPTextEncode."""
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


# ---------------------------------------------------------------------------
# caches: one model config, one clip config, a few VAEs / upscale models
# ---------------------------------------------------------------------------

_MODEL_CACHE = {"key": None, "model": None}
_CLIP_CACHE = {"key": None, "clip": None}
_LORA_SD_CACHE = {}
_LORA_SD_CACHE_MAX = 4
_VAE_CACHE = {}
_VAE_CACHE_MAX = 4
_UPSCALE_MODEL_CACHE = {}
_UPSCALE_MODEL_CACHE_MAX = 2


def _load_lora_state_dict(lora_path):
    if lora_path not in _LORA_SD_CACHE:
        if len(_LORA_SD_CACHE) >= _LORA_SD_CACHE_MAX:
            _LORA_SD_CACHE.clear()
        _LORA_SD_CACHE[lora_path] = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
    return _LORA_SD_CACHE[lora_path]


def _get_model(base_model, weight_dtype, lora_stack):
    """Load (or fetch from cache) the base model with the LoRA stack applied."""
    key = (base_model, weight_dtype, lora_stack)
    if _MODEL_CACHE["key"] == key and _MODEL_CACHE["model"] is not None:
        print("[LTXV Sulphur AIO] model cache hit")
        return _MODEL_CACHE["model"]

    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
    print(f"[LTXV Sulphur AIO] loading base model: {base_model}")
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    for lora_name, strength in lora_stack:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, lora_metadata = _load_lora_state_dict(lora_path)
        print(f"[LTXV Sulphur AIO] applying LoRA: {lora_name} @ {strength}")
        model = comfy.sd.load_lora_for_models(
            model, None, lora_sd, strength, 0, lora_metadata
        )[0]

    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["model"] = model
    return model


_OVERRIDE_MODEL_CACHE = {"key": None, "model": None}


def _apply_lora_stack(model, lora_stack):
    """Apply a LoRA stack to an already-loaded model (used for model_override)."""
    key = (id(model), lora_stack)
    if _OVERRIDE_MODEL_CACHE["key"] == key and _OVERRIDE_MODEL_CACHE["model"] is not None:
        print("[LTXV Sulphur AIO] override-model cache hit")
        return _OVERRIDE_MODEL_CACHE["model"]

    out_model = model
    for lora_name, strength in lora_stack:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, lora_metadata = _load_lora_state_dict(lora_path)
        print(f"[LTXV Sulphur AIO] applying LoRA to model_override: {lora_name} @ {strength}")
        out_model = comfy.sd.load_lora_for_models(
            out_model, None, lora_sd, strength, 0, lora_metadata
        )[0]

    _OVERRIDE_MODEL_CACHE["key"] = key
    _OVERRIDE_MODEL_CACHE["model"] = out_model
    return out_model


def _get_clip(clip_1, clip_2):
    """DualCLIPLoader equivalent, type fixed to ltxv (as in the workflow)."""
    key = (clip_1, clip_2)
    if _CLIP_CACHE["key"] == key and _CLIP_CACHE["clip"] is not None:
        print("[LTXV Sulphur AIO] clip cache hit")
        return _CLIP_CACHE["clip"]

    clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_1)
    clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_2)
    print(f"[LTXV Sulphur AIO] loading text encoders: {clip_1} + {clip_2}")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.LTXV,
        model_options={},
    )
    _CLIP_CACHE["key"] = key
    _CLIP_CACHE["clip"] = clip
    return clip


def _get_vae(vae_name):
    """VAELoader equivalent for regular VAE checkpoints from models/vae."""
    if vae_name in _VAE_CACHE:
        return _VAE_CACHE[vae_name]
    if len(_VAE_CACHE) >= _VAE_CACHE_MAX:
        _VAE_CACHE.clear()

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    print(f"[LTXV Sulphur AIO] loading VAE: {vae_name}")
    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    _VAE_CACHE[vae_name] = vae
    return vae


def _get_upscale_model(model_name):
    """LatentUpscaleModelLoader equivalent, cached."""
    if model_name in _UPSCALE_MODEL_CACHE:
        return _UPSCALE_MODEL_CACHE[model_name]
    if len(_UPSCALE_MODEL_CACHE) >= _UPSCALE_MODEL_CACHE_MAX:
        _UPSCALE_MODEL_CACHE.clear()

    print(f"[LTXV Sulphur AIO] loading latent upscale model: {model_name}")
    upscale_model = _res(LatentUpscaleModelLoader.execute(model_name))[0]
    _UPSCALE_MODEL_CACHE[model_name] = upscale_model
    return upscale_model


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

class LTXVSulphurAllInOne:
    CATEGORY = "LTXV"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "frame_rate")
    DESCRIPTION = (
        "All-in-one LTXV two-pass sampler (Sulphur workflow port). "
        "Pass 1 at half resolution -> 2x latent upscale -> pass 2 at full "
        "resolution. T2V / I2V / I2V+Audio, HD/FHD presets with "
        "ratio-from-image, model+LoRA+CLIP+VAE caching built in."
    )

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        ratio_list = list(HD_RATIOS.keys())
        return {
            "required": {
                "mode": (["▶️ text_to_video", "▶️ image_to_video", "▶️ image_audio_to_video"],
                         {"default": "▶️ image_to_video",
                          "tooltip": "text_to_video: prompt only. image_to_video: connect an image. "
                                     "image_audio_to_video: connect an image AND an audio file."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "What you want to see. LTXV likes detailed, "
                                                          "film-style descriptions with timestamps."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE,
                                               "tooltip": "What to avoid. Default is the negative prompt "
                                                          "from the original workflow."}),
                "base_model": (folder_paths.get_filename_list("diffusion_models"),
                               {"tooltip": "LTXV 2.3 A/V checkpoint from models/diffusion_models "
                                           "(e.g. sulphur2Base). Reloaded only when the selection changes."}),
                "clip_1": (folder_paths.get_filename_list("text_encoders"),
                           {"tooltip": "Main text encoder from models/text_encoders "
                                       "(e.g. gemma-3-12b ... int4)."}),
                "clip_2": (folder_paths.get_filename_list("text_encoders"),
                           {"tooltip": "LTXV text projection from models/text_encoders "
                                       "(e.g. ltx-2.3_text_projection)."}),
                "vae": (folder_paths.get_filename_list("vae"),
                        {"tooltip": "Video VAE from models/vae (e.g. LTX23_video_vae)."}),
                "audio_vae": (folder_paths.get_filename_list("vae"),
                              {"tooltip": "Audio VAE from models/vae (e.g. LTX23_audio_vae)."}),
                "upscale_model": (folder_paths.get_filename_list("latent_upscale_models"),
                                  {"tooltip": "Latent upscaler from models/latent_upscale_models "
                                              "(e.g. ltx-2.3-spatial-upscaler-x2). Used between the passes."}),
                "video_size": (["HD", "FHD", "Custom"], {"default": "HD",
                               "tooltip": "HD ~1280px, FHD ~1920px (same tables as the Star LTX Video "
                                          "Settings node), Custom = custom_width/height below."}),
                "ratio": (ratio_list, {"default": "1:1",
                          "tooltip": "Aspect ratio. Overridden by the input image's ratio when "
                                     "'ratio_from_image' is enabled and an image is connected."}),
                "ratio_from_image": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled",
                                     "tooltip": "Pick the closest preset ratio to the connected image. "
                                                "Falls back to 'ratio' when no image is connected."}),
                "custom_width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32,
                                 "tooltip": "Only used when video_size = Custom."}),
                "custom_height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32,
                                  "tooltip": "Only used when video_size = Custom."}),
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 120, "step": 1,
                               "tooltip": "Frames per second of the output video."}),
                "seconds": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1,
                            "tooltip": "Video length in seconds. Frame count is snapped to 8n+1 "
                                       "(4s @ 25fps = 97 frames)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Shared by both sampling passes."}),
                "sigma_preset": (["12 steps", "8 steps", "16 steps", "custom"],
                                 {"tooltip": "First-pass noise schedule - the three presets from the "
                                             "original workflow's note node. 12 = default, 8 = faster, "
                                             "16 = finer. 'custom' uses custom_sigmas_pass1 below."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Start frame / guide image (image_to_video modes)."}),
                "audio": ("AUDIO", {"tooltip": "Voice / music track (image_audio_to_video mode). "
                                               "Trimmed to the video length and preserved as-is."}),
                "override_audio": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                    "tooltip": "text_to_video / image_to_video only: when disabled (default), "
                                               "the connected 'audio' input is ignored and the model-generated "
                                               "audio is sent to the audio output. When enabled, the connected "
                                               "'audio' input is passed straight to the audio output instead. "
                                               "Ignored in image_audio_to_video mode, where the connected audio "
                                               "is always passed through to the output."}),
                "lora_1": (lora_list, {"tooltip": "Optional LoRA stack, applied in order 1 -> 3."}),
                "lora_1_strength": ("FLOAT", {"default": 0.6, "min": -100.0, "max": 100.0, "step": 0.01,
                                    "tooltip": "The distilled LoRA in the original workflow ran at 0.6."}),
                "lora_2": (lora_list,),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_3": (lora_list,),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "custom_sigmas_pass1": ("STRING", {"multiline": True, "default": SIGMA_PRESETS["12 steps"],
                                        "tooltip": "Only used when sigma_preset = custom."}),
                "sigmas_pass2": ("STRING", {"multiline": False, "default": DEFAULT_PASS2_SIGMAS,
                                 "tooltip": "Second-pass (refine) schedule. Default from the workflow."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                        "tooltip": "Both passes. 1.0 for distilled models, as in the workflow."}),
                "sampler_pass1": (comfy.samplers.SAMPLER_NAMES, {"default": "euler_ancestral_cfg_pp",
                                  "tooltip": "Sampler for pass 1 (half resolution)."}),
                "sampler_pass2": (comfy.samplers.SAMPLER_NAMES, {"default": "euler_cfg_pp",
                                  "tooltip": "Sampler for pass 2 (full resolution refine)."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"tooltip": "Override base-model dtype. 'default' = as stored."}),
                "model_override": ("MODEL", {"tooltip": "Optional external model (e.g. patched with "
                                    "flash/sage attention). When connected, this is used instead of "
                                    "loading 'base_model' from the dropdown, and the LoRA stack below "
                                    "is applied to it directly."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(
        self,
        mode,
        positive_prompt,
        negative_prompt,
        base_model,
        clip_1,
        clip_2,
        vae,
        audio_vae,
        upscale_model,
        video_size,
        ratio,
        ratio_from_image,
        custom_width,
        custom_height,
        frame_rate,
        seconds,
        seed,
        sigma_preset,
        image=None,
        audio=None,
        override_audio=False,
        lora_1="None",
        lora_1_strength=1.0,
        lora_2="None",
        lora_2_strength=1.0,
        lora_3="None",
        lora_3_strength=1.0,
        custom_sigmas_pass1="",
        sigmas_pass2=DEFAULT_PASS2_SIGMAS,
        cfg=1.0,
        sampler_pass1="euler_ancestral_cfg_pp",
        sampler_pass2="euler_cfg_pp",
        weight_dtype="default",
        model_override=None,
        unique_id=None,
    ):

        start_time = time.time()
        event_cb = make_event_cb(unique_id)
        mode = mode.split(" ", 1)[-1] if " " in mode else mode
        use_image = mode in ("image_to_video", "image_audio_to_video")
        use_audio = mode == "image_audio_to_video"
        if use_image and image is None:
            raise ValueError(f"[LTXV Sulphur AIO] mode '{mode}' needs an image input")
        if use_audio and audio is None:
            raise ValueError(f"[LTXV Sulphur AIO] mode '{mode}' needs an audio input")

        # ---- resolution / frames (Star LTX Video Settings logic) -----------
        width, height, size_source = _resolve_size(
            video_size, ratio, ratio_from_image, custom_width, custom_height, image
        )
        frames = _round_8_plus_1(frame_rate * seconds + 1)  # 8n+1
        # pass 1 runs at half resolution (the a/2 math nodes in the workflow)
        w1 = max(64, int(width / 2))
        h1 = max(64, int(height / 2))

        # ---- load models (cached) -------------------------------------------
        lora_stack = tuple(
            (name, strength)
            for name, strength in (
                (lora_1, lora_1_strength),
                (lora_2, lora_2_strength),
                (lora_3, lora_3_strength),
            )
            if name and name != "None" and strength != 0.0
        )
        if model_override is not None:
            print("[LTXV Sulphur AIO] using external model_override instead of base_model dropdown")
            model = _apply_lora_stack(model_override, lora_stack)
        else:
            model = _get_model(base_model, weight_dtype, lora_stack)
        clip = _get_clip(clip_1, clip_2)
        video_vae = _get_vae(vae)
        audio_vae_model = _get_vae(audio_vae)
        upscale = _get_upscale_model(upscale_model)

        # ---- prompts ---------------------------------------------------------
        positive = _encode_prompt(clip, positive_prompt)
        negative = _encode_prompt(clip, negative_prompt)

        # ---- sigma schedules --------------------------------------------------
        sigmas1_text = custom_sigmas_pass1 if sigma_preset == "custom" else SIGMA_PRESETS[sigma_preset]
        sigmas1 = _parse_sigmas(sigmas1_text)
        sigmas2 = _parse_sigmas(sigmas_pass2)

        print(f"[LTXV Sulphur AIO] mode={mode} | {width}x{height} ({size_source}) | "
              f"{frames} frames @ {frame_rate} fps | "
              f"pass1 {len(sigmas1) - 1} steps / pass2 {len(sigmas2) - 1} steps")

        # ---- conditioning: LTXV frame rate -----------------------------------
        pos, neg = _res(LTXVConditioning.execute(positive, negative, float(frame_rate)))

        # ---- pass 1 latents (half res) ---------------------------------------
        video_latent = _res(EmptyLTXVLatentVideo.execute(w1, h1, frames, 1))[0]

        img_prep = None
        if use_image:
            img_resized = _resize_longer_edge(image, GUIDE_RESIZE_LONG_EDGE)
            img_prep = _res(LTXVPreprocess.execute(img_resized, IMG_COMPRESSION))[0]
            video_latent = _res(
                LTXVImgToVideoInplace.execute(video_vae, img_prep, video_latent, 0.7, False)
            )[0]

        if use_audio:
            audio_trim = _res(TrimAudioDuration.execute(audio, 0.0, frames / frame_rate))[0]
            audio_latent = _res(LTXVAudioVAEEncode.execute(audio_trim, audio_vae_model))[0]
            # Keep the encoded audio untouched through both passes
            # (SolidMask value=0 -> SetLatentNoiseMask in the original subgraph).
            audio_latent = audio_latent.copy()
            s = audio_latent["samples"]
            audio_latent["noise_mask"] = torch.zeros(
                (s.shape[0], 1, s.shape[2], s.shape[3]), dtype=torch.float32, device=s.device
            )
        else:
            audio_latent = _res(LTXVEmptyLatentAudio.execute(frames, frame_rate, 1, audio_vae_model))[0]

        av_latent = _res(LTXVConcatAVLatent.execute(video_latent, audio_latent))[0]

        # ---- pass 1 sampling ---------------------------------------------------
        noise = _res(RandomNoise.execute(seed))[0]
        sampler1 = _res(KSamplerSelect.execute(sampler_pass1))[0]

        # Patch model for fancy DOM progress bar (pass 1)
        _rep1 = None
        _cln1 = None
        if event_cb is not None:
            model, _rep1, _cln1 = patch_model_for_progress(
                model, len(sigmas1) - 1, event_cb, is_flux=True, label="pass 1")

        guider1 = _res(CFGGuider.execute(model, pos, neg, cfg))[0]
        out1 = _res(SamplerCustomAdvanced.execute(noise, guider1, sampler1, sigmas1, av_latent))[0]

        if _cln1 is not None:
            _cln1()
        if _rep1 is not None:
            _rep1.finish_unit()

        # ---- middle: split, 2x latent upscale, guide re-injection --------------
        vid1, aud1 = _res(LTXVSeparateAVLatent.execute(out1))
        up = _res(LTXVLatentUpsampler.execute(vid1, upscale, video_vae))[0]
        if use_image:
            up = _res(
                LTXVImgToVideoInplace.execute(video_vae, img_prep, up, 1.0, False)
            )[0]
        pos2, neg2, _ = _res(LTXVCropGuides.execute(pos, neg, vid1))
        av2 = _res(LTXVConcatAVLatent.execute(up, aud1))[0]

        # ---- pass 2 sampling (full res) ----------------------------------------
        sampler2 = _res(KSamplerSelect.execute(sampler_pass2))[0]

        # Patch model for fancy DOM progress bar (pass 2)
        _rep2 = None
        _cln2 = None
        if event_cb is not None:
            model, _rep2, _cln2 = patch_model_for_progress(
                model, len(sigmas2) - 1, event_cb, is_flux=True, label="pass 2")

        guider2 = _res(CFGGuider.execute(model, pos2, neg2, cfg))[0]
        out2 = _res(SamplerCustomAdvanced.execute(noise, guider2, sampler2, sigmas2, av2))[0]

        if _cln2 is not None:
            _cln2()
        if _rep2 is not None:
            _rep2.finish_unit()

        # ---- decode --------------------------------------------------------------
        vid2, aud2 = _res(LTXVSeparateAVLatent.execute(out2))
        images = nodes.VAEDecode().decode(video_vae, vid2)[0]
        audio_out = _res(LTXVAudioVAEDecode.execute(aud2, audio_vae_model))[0]

        # ---- audio output selection ----------------------------------------
        # image_audio_to_video: connected audio always passes through.
        # otherwise: pass through only if override_audio is enabled and audio is connected,
        # else use the model-generated audio (default behaviour).
        if use_audio:
            audio_out = audio
        elif override_audio and audio is not None:
            audio_out = audio

        if _rep2 is not None:
            _rep2.finish_all(time.time() - start_time)

        return (images, audio_out, float(frame_rate))


NODE_CLASS_MAPPINGS = {
    "LTXVSulphurAllInOne": LTXVSulphurAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSulphurAllInOne": "⭐ Star LTXV All-in-One (2-Pass)",
}
