"""
LTXV 2.5 Sulphur All-in-One engine node.

Single-node port of the official LTXV 2.5 workflow templates, using a single
text encoder of type ltxv:
  - text_to_video / image_to_video / image_audio_to_video: two-pass pipeline
    (pass 1 at half resolution -> 2x latent upscale -> pass 2 at full res),
  - first_last_frame_to_video: single full-res pass with the first and last
    frame added as keyframe guides (LTXVAddGuide, like the flf2v template),
  - audio_only: one 30-step pass at 64x64 with a plain normal schedule, only
    the audio output matters,
with baked sigma presets (8 / 12 / 16 steps) or plain 20 / 30 / 40 / 50-step
normal schedules, HD/FHD size presets with ratio-from-image (Star LTX Video
Settings logic), base model + up to three LoRAs, VAE / audio-VAE / CLIP /
latent-upscaler dropdowns, internal prompt encoding - and internal caches
so big files are only re-loaded when the selection actually changes.

Generated audio is always decoded from the first (high-step) sampling pass,
never from the upscale refine pass.

Nothing else is required on the canvas except LoadImage / LoadAudio for the
i2v modes and your save / upscale nodes downstream.
"""

import re
import time

import math

import numpy as np
import torch
from PIL import Image

import folder_paths
import nodes
import comfy.sd
import comfy.samplers
import comfy.utils
import comfy.model_management  # noqa: F401  (kept for parity with loader nodes)

from ..misc.star_progress import make_event_cb, patch_model_for_progress
from .star_video_sound_enricher import process_audio as _enrich_sound

from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVImgToVideoInplace,
    LTXVAddGuide,
    LTXVConditioning,
    LTXVCropGuides,
    LTXVPreprocess,
    LTXVConcatAVLatent,
    LTXVSeparateAVLatent,
)

try:
    from comfy_extras.nodes_lt import LTXVDualCFGGuider
except ImportError:
    # Older ComfyUI without the LTXV-AV dual-CFG guider node: vendored copy
    # (identical logic to the core Guider_LTXAVDualCFG).
    class _GuiderLTXAVDualCFG(comfy.samplers.CFGGuider):
        """CFG guider with separate scales for the video/audio modalities of
        a packed LTXV-AV latent."""

        def set_conds(self, positive, negative):
            self.inner_set_conds({"positive": positive, "negative": negative})

        def set_cfg(self, video_cfg, audio_cfg):
            self.video_cfg = video_cfg
            self.audio_cfg = audio_cfg
            self.cfg = max(video_cfg, audio_cfg)

        def sample(self, noise, latent_image, *args, **kwargs):
            self._v_numel = None
            if getattr(latent_image, "is_nested", False):
                parts = latent_image.unbind()
                if len(parts) >= 2:
                    self._v_numel = math.prod(parts[0].shape[1:])
            return super().sample(noise, latent_image, *args, **kwargs)

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            v = getattr(self, "_v_numel", None)
            if v is None or math.isclose(self.video_cfg, self.audio_cfg):
                self.cfg = self.video_cfg
                return super().predict_noise(x, timestep, model_options, seed)

            video_cfg, audio_cfg = self.video_cfg, self.audio_cfg

            def dual_cfg(args):
                cond, uncond = args["cond"], args["uncond"]
                out = uncond + (cond - uncond) * video_cfg
                out[..., v:] = uncond[..., v:] + (cond[..., v:] - uncond[..., v:]) * audio_cfg
                return out

            model_options = {**model_options, "sampler_cfg_function": dual_cfg,
                             "disable_cfg1_optimization": True}
            return super().predict_noise(x, timestep, model_options, seed)

    class LTXVDualCFGGuider:
        @classmethod
        def execute(cls, model, positive, negative, video_cfg, audio_cfg):
            guider = _GuiderLTXAVDualCFG(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(video_cfg, audio_cfg)
            return (guider,)
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
    BasicScheduler,
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
DEFAULT_PASS2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

# Plain-step presets: normal scheduler instead of a baked sigma list.
STEP_PRESETS = ("20 steps", "30 steps", "40 steps", "50 steps")

# Audio-only pass: 30 steps, normal scheduler, minimal 64x64 video.
AUDIO_STEPS = 30
AUDIO_ONLY_SIZE = 64

# First/last-frame guide strength used by the official flf2v template.
FLF_GUIDE_STRENGTH = 0.7

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
        raise ValueError("[LTXV 2.5 AIO] sigma schedule needs at least two values")
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


def _resize_to(image, width, height):
    """Center-crop resize to exact pixel dims (flf2v template's ResizeImages)."""
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)


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
        print("[LTXV 2.5 AIO] model cache hit")
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
    print(f"[LTXV 2.5 AIO] loading base model: {base_model}")
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    for lora_name, strength in lora_stack:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, lora_metadata = _load_lora_state_dict(lora_path)
        print(f"[LTXV 2.5 AIO] applying LoRA: {lora_name} @ {strength}")
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
        print("[LTXV 2.5 AIO] override-model cache hit")
        return _OVERRIDE_MODEL_CACHE["model"]

    out_model = model
    for lora_name, strength in lora_stack:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, lora_metadata = _load_lora_state_dict(lora_path)
        print(f"[LTXV 2.5 AIO] applying LoRA to model_override: {lora_name} @ {strength}")
        out_model = comfy.sd.load_lora_for_models(
            out_model, None, lora_sd, strength, 0, lora_metadata
        )[0]

    _OVERRIDE_MODEL_CACHE["key"] = key
    _OVERRIDE_MODEL_CACHE["model"] = out_model
    return out_model


def _get_clip(clip_1):
    """Single CLIPLoader equivalent, type fixed to ltxv (LTXV 2.5 uses one encoder)."""
    key = (clip_1,)
    if _CLIP_CACHE["key"] == key and _CLIP_CACHE["clip"] is not None:
        print("[LTXV 2.5 AIO] clip cache hit")
        return _CLIP_CACHE["clip"]

    clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_1)
    print(f"[LTXV 2.5 AIO] loading text encoder: {clip_1}")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1],
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
    print(f"[LTXV 2.5 AIO] loading VAE: {vae_name}")
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

    print(f"[LTXV 2.5 AIO] loading latent upscale model: {model_name}")
    upscale_model = _res(LatentUpscaleModelLoader.execute(model_name))[0]
    _UPSCALE_MODEL_CACHE[model_name] = upscale_model
    return upscale_model


def _main_sigmas(sigma_preset, custom_sigmas_pass1, model):
    """First/main pass schedule: baked preset, plain N-step schedule, or custom."""
    if sigma_preset == "custom":
        return _parse_sigmas(custom_sigmas_pass1)
    if sigma_preset in SIGMA_PRESETS:
        return _parse_sigmas(SIGMA_PRESETS[sigma_preset])
    steps = int(sigma_preset.split(" ", 1)[0])  # "20 steps" -> 20
    return _res(BasicScheduler.execute(model, "normal", steps, 1.0))[0]


def _run_audio_pass(model, pos, neg, frames, frame_rate, seed, cfg,
                    sampler_name, audio_vae_model, event_cb, start_time, label):
    """Single 64x64 A/V pass on a plain 30-step schedule - best-effort soundtrack.

    Used by the audio_only mode, always with the same model as the video so
    the soundtrack matches. Returns (decoded_audio, video_latent, av_latent)
    so the caller can still decode the tiny video as a visual reference and
    expose the sampled latent as an output.
    """
    video_latent = _res(EmptyLTXVLatentVideo.execute(
        AUDIO_ONLY_SIZE, AUDIO_ONLY_SIZE, frames, 1))[0]
    audio_latent = _res(LTXVEmptyLatentAudio.execute(
        frames, frame_rate, 1, audio_vae_model))[0]
    av_latent = _res(LTXVConcatAVLatent.execute(video_latent, audio_latent))[0]
    sigmas = _res(BasicScheduler.execute(model, "normal", AUDIO_STEPS, 1.0))[0]
    noise = _res(RandomNoise.execute(seed))[0]
    sampler = _res(KSamplerSelect.execute(sampler_name))[0]

    rep = None
    cln = None
    if event_cb is not None:
        model, rep, cln = patch_model_for_progress(
            model, AUDIO_STEPS, event_cb, is_flux=True, label=label)
    guider = _res(LTXVDualCFGGuider.execute(model, pos, neg, cfg, cfg))[0]
    out = _res(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av_latent))[0]
    if cln is not None:
        cln()

    vid, aud = _res(LTXVSeparateAVLatent.execute(out))
    audio = _res(LTXVAudioVAEDecode.execute(aud, audio_vae_model))[0]
    if rep is not None:
        rep.finish_all(time.time() - start_time)
    return audio, vid, out


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

class LTXV25SulphurAllInOne:
    CATEGORY = "⭐StarNodes/Video"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "LATENT", "MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("images", "audio", "frame_rate", "latent", "model", "clip", "vae", "audio_vae")
    DESCRIPTION = (
        "All-in-one LTXV 2.5 sampler (official template port). T2V / I2V / "
        "I2V+Audio run two passes (half res -> 2x latent upscale -> full res), "
        "First/Last-Frame runs a single full-res pass with keyframe guides, "
        "Audio Only renders just the soundtrack (64x64, 30 steps). Generated "
        "audio is always decoded from the first (high-step) pass. "
        "Model+LoRA+CLIP+VAE caching built in. Uses a single ltxv text "
        "encoder (LTXV 2.5). Also outputs the sampled latent and the loaded "
        "model, clip and both VAEs for downstream reuse."
    )

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        ratio_list = list(HD_RATIOS.keys())
        return {
            "required": {
                "mode": (["▶️ text_to_video", "▶️ image_to_video", "▶️ image_audio_to_video",
                          "▶️ first_last_frame_to_video", "🎵 audio_only"],
                         {"default": "▶️ image_to_video",
                          "tooltip": "text_to_video: prompt only. image_to_video: connect first_frame. "
                                     "image_audio_to_video: connect first_frame AND an audio file. "
                                     "first_last_frame_to_video: connect first_frame AND last_frame "
                                     "(single full-res pass with keyframe guides). "
                                     "audio_only: no real video - one 30-step pass at 64x64, "
                                     "only the audio output matters."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "What you want to see. LTXV likes detailed, "
                                                          "film-style descriptions with timestamps."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE,
                                               "tooltip": "What to avoid. Default is the negative prompt "
                                                          "from the original workflow."}),
                "base_model": (folder_paths.get_filename_list("diffusion_models"),
                               {"default": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                                "tooltip": "LTXV 2.5 A/V checkpoint from models/diffusion_models. "
                                           "Reloaded only when the selection changes."}),
                "clip_1": (folder_paths.get_filename_list("text_encoders"),
                           {"default": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                            "tooltip": "Single LTXV 2.5 text encoder from models/text_encoders "
                                       "(type ltxv)."}),
                "vae": (folder_paths.get_filename_list("vae"),
                        {"default": "ltx-2.5-video-vae-conv-bf16.safetensors",
                         "tooltip": "Video VAE from models/vae (e.g. ltx-2.5-video-vae-conv-bf16)."}),
                "audio_vae": (folder_paths.get_filename_list("vae"),
                              {"default": "ltx-2.5-audio-vae-bf16.safetensors",
                               "tooltip": "Audio VAE from models/vae (e.g. ltx-2.5-audio-vae-bf16)."}),
                "upscale_model": (folder_paths.get_filename_list("latent_upscale_models"),
                                  {"default": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                                   "tooltip": "Latent upscaler from models/latent_upscale_models "
                                              "(e.g. ltx-2.5-latent-spatial-upscaler-x2). "
                                              "Used between the passes."}),
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
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1,
                               "tooltip": "Frames per second of the output video."}),
                "seconds": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1,
                            "tooltip": "Video length in seconds. Frame count is snapped to 8n+1 "
                                       "(4s @ 25fps = 97 frames)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Shared by both sampling passes."}),
                "sigma_preset": (["12 steps", "8 steps", "16 steps", *STEP_PRESETS, "custom"],
                                 {"tooltip": "Main-pass noise schedule. 8/12/16 = the baked schedules "
                                             "from the original workflow's note node (12 = default, "
                                             "8 = faster, 16 = finer). 20/30/40/50 = plain sampler "
                                             "steps with the normal scheduler, no custom sigmas. "
                                             "'custom' uses custom_sigmas_pass1 below."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Start frame / guide image (image_to_video, "
                                "image_audio_to_video and first_last_frame_to_video modes)."}),
                "last_frame": ("IMAGE", {"tooltip": "Last frame (first_last_frame_to_video mode only). "
                               "Center-crop resized to the video size and added as the final keyframe."}),
                "audio": ("AUDIO", {"tooltip": "Voice / music track (image_audio_to_video mode only). "
                                               "Trimmed to the video length and preserved as-is. "
                                               "Ignored in all other modes."}),

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
                "sampler_pass1": (comfy.samplers.SAMPLER_NAMES, {"default": "euler_ancestral",
                                  "tooltip": "Sampler for pass 1 (half resolution)."}),
                "sampler_pass2": (comfy.samplers.SAMPLER_NAMES, {"default": "euler_ancestral",
                                  "tooltip": "Sampler for pass 2 (full resolution refine)."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"tooltip": "Override base-model dtype. 'default' = as stored."}),
                "model_override": ("MODEL", {"tooltip": "Optional external model (e.g. patched with "
                                    "flash/sage attention). When connected, this is used instead of "
                                    "loading 'base_model' from the dropdown, and the LoRA stack below "
                                    "is applied to it directly."}),
                "sound_settings": ("SOUND_SETTINGS", {"tooltip": "Optional sound processing from a "
                                   "'Star Video Sound Enricher Option' node - the audio output is "
                                   "cleaned up and enriched with these settings (at least 44.1 kHz, "
                                   "never downsampled) before it leaves the node."}),
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
        first_frame=None,
        last_frame=None,
        audio=None,
        lora_1="None",
        lora_1_strength=1.0,
        lora_2="None",
        lora_2_strength=1.0,
        lora_3="None",
        lora_3_strength=1.0,
        custom_sigmas_pass1="",
        sigmas_pass2=DEFAULT_PASS2_SIGMAS,
        cfg=1.0,
        sampler_pass1="euler_ancestral",
        sampler_pass2="euler_ancestral",
        weight_dtype="default",
        model_override=None,
        sound_settings=None,
        unique_id=None,
    ):

        start_time = time.time()
        event_cb = make_event_cb(unique_id)
        mode = mode.split(" ", 1)[-1] if " " in mode else mode
        is_flf = mode == "first_last_frame_to_video"
        is_audio_only = mode == "audio_only"
        use_image = mode in ("image_to_video", "image_audio_to_video", "first_last_frame_to_video")
        use_audio = mode == "image_audio_to_video"
        if use_image and first_frame is None:
            raise ValueError(f"[LTXV 2.5 AIO] mode '{mode}' needs a first_frame input")
        if is_flf and last_frame is None:
            raise ValueError(f"[LTXV 2.5 AIO] mode '{mode}' needs a last_frame input")
        if use_audio and audio is None:
            raise ValueError(f"[LTXV 2.5 AIO] mode '{mode}' needs an audio input")

        # ---- resolution / frames (Star LTX Video Settings logic) -----------
        width, height, size_source = _resolve_size(
            video_size, ratio, ratio_from_image, custom_width, custom_height, first_frame
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
            print("[LTXV 2.5 AIO] using external model_override instead of base_model dropdown")
            model = _apply_lora_stack(model_override, lora_stack)
        else:
            model = _get_model(base_model, weight_dtype, lora_stack)
        model_out = model  # keep an unpatched reference for the MODEL output
        clip = _get_clip(clip_1)
        video_vae = _get_vae(vae)
        audio_vae_model = _get_vae(audio_vae)
        upscale = None if (is_flf or is_audio_only) else _get_upscale_model(upscale_model)

        # ---- prompts ---------------------------------------------------------
        positive = _encode_prompt(clip, positive_prompt)
        negative = _encode_prompt(clip, negative_prompt)

        # ---- conditioning: LTXV frame rate -----------------------------------
        pos, neg = _res(LTXVConditioning.execute(positive, negative, float(frame_rate)))

        # ---- main-pass sigma schedule ------------------------------------------
        sigmas1 = _main_sigmas(sigma_preset, custom_sigmas_pass1, model)

        print(f"[LTXV 2.5 AIO] mode={mode} | {width}x{height} ({size_source}) | "
              f"{frames} frames @ {frame_rate} fps")

        # ---- audio only: one plain 30-step pass at 64x64 ----------------------
        if is_audio_only:
            print(f"[LTXV 2.5 AIO] audio-only pass: {AUDIO_STEPS} steps @ "
                  f"{AUDIO_ONLY_SIZE}x{AUDIO_ONLY_SIZE}")
            audio_out, vid, av_out = _run_audio_pass(
                model, pos, neg, frames, frame_rate, seed, cfg,
                sampler_pass1, audio_vae_model, event_cb, start_time, "audio only")
            images = nodes.VAEDecodeTiled().decode(video_vae, vid, 768, 64, 4096, 32)[0]
            if sound_settings is not None:
                audio_out = _enrich_sound(audio_out, sound_settings)
            return (images, audio_out, float(frame_rate), av_out,
                    model_out, clip, video_vae, audio_vae_model)

        if is_flf:
            # ---- first/last frame to video: single full-res pass with guides ---
            print(f"[LTXV 2.5 AIO] flf2v single pass | {len(sigmas1) - 1} steps")
            video_latent = _res(EmptyLTXVLatentVideo.execute(width, height, frames, 1))[0]
            first_prep = _res(LTXVPreprocess.execute(
                _resize_to(first_frame, width, height), IMG_COMPRESSION))[0]
            last_prep = _res(LTXVPreprocess.execute(
                _resize_to(last_frame, width, height), IMG_COMPRESSION))[0]
            pos_g, neg_g, video_latent = _res(LTXVAddGuide.execute(
                pos, neg, video_vae, video_latent, first_prep, 0, FLF_GUIDE_STRENGTH))
            pos_g, neg_g, video_latent = _res(LTXVAddGuide.execute(
                pos_g, neg_g, video_vae, video_latent, last_prep, -1, FLF_GUIDE_STRENGTH))
            audio_latent = _res(LTXVEmptyLatentAudio.execute(frames, frame_rate, 1, audio_vae_model))[0]
            av_latent = _res(LTXVConcatAVLatent.execute(video_latent, audio_latent))[0]

            noise = _res(RandomNoise.execute(seed))[0]
            sampler1 = _res(KSamplerSelect.execute(sampler_pass1))[0]

            # Patch model for fancy DOM progress bar
            _rep1 = None
            _cln1 = None
            if event_cb is not None:
                model, _rep1, _cln1 = patch_model_for_progress(
                    model, len(sigmas1) - 1, event_cb, is_flux=True, label="sampling")

            guider1 = _res(LTXVDualCFGGuider.execute(model, pos_g, neg_g, cfg, cfg))[0]
            out1 = _res(SamplerCustomAdvanced.execute(noise, guider1, sampler1, sigmas1, av_latent))[0]

            if _cln1 is not None:
                _cln1()

            # split, crop the guide keyframes back out, decode
            vid1, aud1 = _res(LTXVSeparateAVLatent.execute(out1))
            _, _, vid1 = _res(LTXVCropGuides.execute(pos_g, neg_g, vid1))
            images = nodes.VAEDecodeTiled().decode(video_vae, vid1, 768, 64, 4096, 64)[0]
            gen_audio = _res(LTXVAudioVAEDecode.execute(aud1, audio_vae_model))[0]
            final_latent = out1
            main_rep = _rep1
        else:
            # ---- two-pass pipeline (t2v / i2v / i2v+audio) -----------------------
            sigmas2 = _parse_sigmas(sigmas_pass2)
            print(f"[LTXV 2.5 AIO] pass1 {len(sigmas1) - 1} steps (half res) / "
                  f"pass2 {len(sigmas2) - 1} steps (full res)")

            # ---- pass 1 latents (half res) ---------------------------------------
            video_latent = _res(EmptyLTXVLatentVideo.execute(w1, h1, frames, 1))[0]

            img_prep = None
            if use_image:
                img_resized = _resize_longer_edge(first_frame, GUIDE_RESIZE_LONG_EDGE)
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

            guider1 = _res(LTXVDualCFGGuider.execute(model, pos, neg, cfg, cfg))[0]
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

            guider2 = _res(LTXVDualCFGGuider.execute(model, pos2, neg2, cfg, cfg))[0]
            out2 = _res(SamplerCustomAdvanced.execute(noise, guider2, sampler2, sigmas2, av2))[0]

            if _cln2 is not None:
                _cln2()
            if _rep2 is not None:
                _rep2.finish_unit()

            # ---- decode --------------------------------------------------------------
            # Audio always comes from the first (high-step) pass, not the refine pass.
            vid2, _ = _res(LTXVSeparateAVLatent.execute(out2))
            images = nodes.VAEDecodeTiled().decode(video_vae, vid2, 768, 64, 4096, 32)[0]
            gen_audio = _res(LTXVAudioVAEDecode.execute(aud1, audio_vae_model))[0]
            final_latent = out2
            main_rep = _rep2

        # ---- audio output selection ---------------------------------------------
        # image_audio_to_video: the connected audio always passes through,
        # otherwise the generated audio (decoded from pass 1) is used.
        audio_out = audio if use_audio else gen_audio

        if sound_settings is not None:
            print("[LTXV 2.5 AIO] applying sound enricher settings to the audio output")
            audio_out = _enrich_sound(audio_out, sound_settings)

        if main_rep is not None:
            main_rep.finish_all(time.time() - start_time)

        return (images, audio_out, float(frame_rate), final_latent,
                model_out, clip, video_vae, audio_vae_model)


NODE_CLASS_MAPPINGS = {
    "LTXV25SulphurAllInOne": LTXV25SulphurAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV25SulphurAllInOne": "⭐ Star LTXV 2.5 All-in-One (BETA)",
}
