"""Star Minimax Latent Upscaler / Star Minimax Latent Upscaler Option

Second-pass latent upscale + refine for MiniMax H3 A/V latents:

  pass-1 LATENT -> video latent upscaled with a MiniMax H3 3D latent-upscaler
  model (models/latent_upscale_models) -> recombined with the pass-1 audio
  latent -> short refine sampling pass (baked 3/4/5-step sigma schedules)
  -> video/audio decode.

Two nodes share the machinery:
  - StarMinimaxLatentUpscaler: standalone LATENT -> IMAGE/AUDIO/FPS/LATENT
    node for any workflow (connect model, clip, video VAE, audio VAE and a
    prompt for the refine pass).
  - StarMinimaxLatentUpscalerOption: same settings as an UPSCALE_SETTINGS
    bundle for the "options" input of the Star Minimax All In One node -
    the refine pass then reuses the pass-1 conditioning (reference latents
    are resolution-matched) and the pass-1 seed.

Self-contained: the 3D upscaler network + loader are vendored from
"Comfyui_Minimax_h3_latent_Upscaler" (MinimaxH3LatentUpscaler3D) and the
reference-conditioning upscale helpers from "ComfyUI-MiniMaxH3_LatentUpscaler",
so neither pack has to be installed.
"""

import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.ldm.common_dit
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from .minimax_common import decode_audio, decode_video, run_sample

_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER))

VAE_DOWNSAMPLE = 16

# Baked refine schedules from the reference workflow's ManualSigmas notes.
SIGMA_PRESETS = {
    "3 steps": "0.9035, 0.6316, 0.3158, 0.0000",
    "4 steps": "0.9035, 0.8000, 0.6316, 0.3158, 0.0000",
    "5 steps": "0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0000",
}

# ---------------------------------------------------------------------------
# Vendored 3D latent upscaler (MinimaxH3LatentUpscaler3D, megapixels mode)
# ---------------------------------------------------------------------------

# MiniMax H3 latent normalization stats (24 channels)
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class _LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, _TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk  # frames of temporal overlap between chunks
        chunk = 24    # effective frames per chunk

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        logging.info("[Star Minimax Latent Upscaler] temporal chunking: T=%d chunks=%d overlap=%d",
                     T, (T + chunk - 1) // chunk, overlap)

        # replicate padding avoids edge flicker on the first/last frames
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)

            # output range includes the overlap so neighbouring chunks blend
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)

            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi]
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)

            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


_UPSCALER_CACHE = {}


def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_found = False
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        if k.endswith('dwconv.weight'):
            temporal_found = True
            cfg["temporal_kernel"] = sd[k].shape[2]

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    if not temporal_found:
        cfg["temporal_every"] = 0
    return cfg


def _load_upscale_model(name, device, precision):
    cache_key = f"{name}::{precision}"
    if cache_key in _UPSCALER_CACHE:
        logging.info("[Star Minimax Latent Upscaler] reusing cached upscale model %s", name)
        return _UPSCALER_CACHE[cache_key].to(device)

    path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    if "conv_in.weight" not in sd:
        raise ValueError(f"[Star Minimax Latent Upscaler] '{name}' is not a MiniMax H3 3D "
                         "latent upscaler model (e.g. minimax_h3_latent_upscaler_3d_fp16.safetensors).")

    cfg = _detect_arch(sd)
    model = _LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"], channels=cfg["channels"],
        dropout=cfg["dropout"], temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"])
    model.load_state_dict(sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    _UPSCALER_CACHE[cache_key] = model
    logging.info("[Star Minimax Latent Upscaler] loaded %s (%s params, temporal=%s, %s)",
                 name, f"{sum(p.numel() for p in model.parameters()):,}",
                 "on" if cfg["temporal_every"] > 0 else "off", precision)
    return model


def upscale_video_latent_3d(video, model_name, megapixels, align,
                            enable_chunking, device_name, precision):
    """Upscale a video latent [B, C, T, H, W] (or [B, C, H, W]) with the 3D
    upscaler model to the megapixel target. Returns (latent on CPU in the
    original dtype, effective scale)."""
    dev = torch.device("cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")
    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    orig_dtype = video.dtype
    was_4d = (video.dim() == 4)
    s = video.to(device=dev, dtype=compute_dtype).clone()
    if was_4d:
        s = s.unsqueeze(2)

    b, c, t, h_in, w_in = s.shape

    # target size in pixel space (aspect locked), aligned, snapped to the VAE grid
    target_pixels = megapixels * 1024 * 1024
    aspect = w_in / h_in
    h_pixel = (target_pixels / aspect) ** 0.5
    w_pixel = h_pixel * aspect
    effective_scale = (w_pixel / (w_in * VAE_DOWNSAMPLE) + h_pixel / (h_in * VAE_DOWNSAMPLE)) / 2.0

    alignment = max(1, align)
    w_pixel = round(round(w_pixel / alignment) * alignment / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE
    h_pixel = round(round(h_pixel / alignment) * alignment / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE
    w_out = max(1, w_pixel // VAE_DOWNSAMPLE)
    h_out = max(1, h_pixel // VAE_DOWNSAMPLE)

    if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise ValueError("[Star Minimax Latent Upscaler] only upscaling is supported "
                         "(effective scale >= 1.0) - raise the megapixels target.")
    if w_out == w_in and h_out == h_in:
        return video, 1.0

    logging.info("[Star Minimax Latent Upscaler] latent %dx%d -> %dx%d (pixels %dx%d, %.2fx)",
                 w_in, h_in, w_out, h_out,
                 w_out * VAE_DOWNSAMPLE, h_out * VAE_DOWNSAMPLE, effective_scale)

    model = _load_upscale_model(model_name, dev, precision)
    mean, std = _make_norm_tensors(dev, compute_dtype)

    s_norm = (s - mean) / std
    out = model(s_norm, scale=effective_scale, target_size=(t, h_out, w_out),
                enable_chunking=enable_chunking)
    del s_norm
    out = out * std + mean

    if was_4d:
        out = out.squeeze(2)
    out = out.to(device="cpu", dtype=orig_dtype)

    # park the upscaler on CPU so the second sampling pass gets the VRAM back
    if dev.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    return out, w_out / w_in


# ---------------------------------------------------------------------------
# Vendored conditioning upscale (upscale_minimax_conditioning) - ref latents
# are bilinearly resolution-matched to the upscaled canvas, audio untouched.
# ---------------------------------------------------------------------------

def _snap_even(size):
    return max(2, ((int(size) + 1) // 2) * 2)


def _upscale_visual_latent(z, scale_by):
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for visual latent, got {type(z)}")
    squeeze = z.ndim == 4
    if squeeze:
        z = z.unsqueeze(2)
    if z.ndim != 5:
        raise ValueError(f"Visual latent needs 4 or 5 dims, got shape {tuple(z.shape)}")

    b, c, t, h, w = z.shape
    h2 = _snap_even(round(h * scale_by))
    w2 = _snap_even(round(w * scale_by))
    flat = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = F.interpolate(flat, size=(h2, w2), mode="bilinear", align_corners=False)
    z = flat.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4)
    z = comfy.ldm.common_dit.pad_to_patch_size(z, (1, 2, 2))
    return z.squeeze(2) if squeeze else z


def _upscale_ref_block(block, scale_by):
    if block.get("kind") == "audio" or block.get("latent") is None:
        return block
    out = dict(block)
    z = _upscale_visual_latent(out["latent"], scale_by)
    out["latent"] = z
    # PackedLayout reads these for RoPE / row counts - keep in sync with the tensor
    out["latent_h"] = int(z.shape[-2])
    out["latent_w"] = int(z.shape[-1])
    if z.ndim == 5 and "latent_t" in out:
        out["latent_t"] = int(z.shape[2])
    return out


def upscale_minimax_conditioning(conditioning, scale_by):
    """Clone CONDITIONING and resolution-match the MiniMax ref latents so the
    pass-1 conditioning can drive the second pass at the upscaled size."""
    if conditioning is None or scale_by == 1.0:
        return conditioning
    out = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            out.append(entry)
            continue
        emb, meta = entry[0], entry[1]
        refs = meta.get("minimax_refs")
        if not refs:
            out.append(entry)
            continue
        new_meta = meta.copy()
        new_meta["minimax_refs"] = [_upscale_ref_block(blk, scale_by) for blk in refs]
        out.append([emb, new_meta])
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _upscale_model_list():
    files = folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
    return files if files else ["(place models in: models/latent_upscale_models)"]


def _first_present(options, preferred):
    return preferred if preferred in options else options[0]


def _parse_sigmas(sigmas_preset):
    return torch.tensor([float(v) for v in SIGMA_PRESETS[sigmas_preset].split(",")],
                        dtype=torch.float32)


def _check_model_name(upscale_model):
    if upscale_model.startswith("("):
        raise ValueError("[Star Minimax Latent Upscaler] no latent upscaler models found - "
                         "place them into models/latent_upscale_models")


def _settings_inputs(upscale_models):
    """The upscale/refine widgets shared by both nodes (fresh instances per call)."""
    return [
        io.Combo.Input("upscale_model", options=upscale_models,
                       default=_first_present(upscale_models, "minimax_h3_latent_upscaler_3d_fp16.safetensors"),
                       tooltip="MiniMax H3 latent upscaler from models/latent_upscale_models."),
        io.Float.Input("megapixels", default=1.0, min=0.1, max=8.0, step=0.1,
                       tooltip="Target total megapixels for the upscaled video (aspect ratio of the pass-1 canvas is kept)."),
        io.Combo.Input("sigmas_preset", options=list(SIGMA_PRESETS.keys()), default="3 steps",
                       tooltip="Refine-pass noise schedule: " +
                               " | ".join(f"{k}: {v}" for k, v in SIGMA_PRESETS.items())),
        io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler",
                       tooltip="Sampler for the refine pass (euler matches the reference workflow, e.g. with a turbo LoRA on the refine model)."),
        io.Boolean.Input("upscale_pass_audio", default=False,
                         label_on="Upscale Pass Audio", label_off="Use 1st pass audio",
                         tooltip="Which pass the audio output is decoded from. 'Use 1st pass audio' keeps the pass-1 soundtrack untouched; 'Upscale Pass Audio' decodes the audio after the refine pass (it is re-noised and rewritten there)."),
        io.Int.Input("align", default=32, min=1, max=512, step=1, advanced=True,
                     tooltip="Pixel-space alignment of the upscaled size. 32 is recommended to avoid light banding."),
        io.Boolean.Input("enable_chunking", default=True, advanced=True,
                         label_on="enabled", label_off="disabled",
                         tooltip="Temporal chunking saves VRAM on long videos. Disable for short clips for pure full-context inference."),
        io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", advanced=True,
                       tooltip="Execution device for the upscaler model (ROCm uses 'cuda')."),
        io.Combo.Input("precision", options=["fp16", "fp32", "bf16"], default="fp16", advanced=True,
                       tooltip="Precision the upscaler model runs at."),
    ]


class StarMinimaxLatentUpscaler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StarMinimaxLatentUpscaler",
            display_name="⭐ Star Minimax Latent Upscaler",
            category="⭐StarNodes/Video",
            description=(
                "Standalone second-pass latent upscale for MiniMax H3: takes the pass-1 "
                "A/V latent from any workflow (e.g. the LATENT output of the Star Minimax "
                "All In One node), upscales the video latent with the selected 3D latent "
                "upscaler model, refines it in a short sampling pass (same seed handling, "
                "baked 3/4/5-step schedules) conditioned on the prompt, and decodes video "
                "+ audio. Works without the All In One node."
            ),
            inputs=[
                io.Latent.Input("latent",
                                tooltip="Pass-1 MiniMax H3 A/V latent (NestedTensor with video + audio)."),
                io.Model.Input("model",
                               tooltip="Diffusion model for the refine pass (e.g. with a turbo LoRA and/or attention patch applied)."),
                io.Clip.Input("clip",
                              tooltip="MiniMax text encoder (qwen3vl) - encodes the prompt for the refine pass."),
                io.Vae.Input("vae",
                             tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae",
                             tooltip="MiniMax H3 audio VAE."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Optional soundtrack passthrough: when the audio toggle is 'Use 1st pass audio', this audio goes straight to the AUDIO output instead of the decoded pass-1 audio (e.g. the original soundtrack of a source video). Ignored when the toggle is 'Upscale Pass Audio'."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Prompt for the refine pass (plain text conditioning; reference tags are not available in the standalone node)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True,
                             tooltip="Noise seed for the refine pass. Use the same seed as pass 1 to reproduce the All In One behavior."),
                *_settings_inputs(_upscale_model_list()),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Audio.Output(display_name="AUDIO"),
                io.Float.Output(display_name="FPS",
                                tooltip="Fixed frame rate of the video (24.0)."),
                io.Latent.Output(display_name="LATENT",
                                 tooltip="The refined second-pass A/V latent, before VAE decoding."),
            ],
        )

    @classmethod
    def execute(cls, latent, model, clip, vae, audio_vae, prompt, seed,
                upscale_model, megapixels, sigmas_preset, sampler_name,
                upscale_pass_audio, align, enable_chunking, device, precision,
                audio=None) -> io.NodeOutput:
        _check_model_name(upscale_model)

        samples_in = latent["samples"]
        if samples_in.is_nested:
            video, audio_member = samples_in.unbind()
        else:
            video, audio_member = samples_in, None

        video_up, scale = upscale_video_latent_3d(
            video, upscale_model, megapixels, align, enable_chunking, device, precision)
        logging.info("[Star Minimax Latent Upscaler] refine pass: %.2fx to %dx%d latent | %s | sampler %s",
                     scale, video_up.shape[-1], video_up.shape[-2], sigmas_preset, sampler_name)

        tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)

        latent2 = latent.copy()
        latent2["samples"] = (comfy.nested_tensor.NestedTensor((video_up.to(audio_member.device), audio_member))
                              if audio_member is not None else video_up)
        samples = run_sample(model, cond, latent2, seed, sampler_name,
                             _parse_sigmas(sigmas_preset))

        images = decode_video(vae, samples)
        if audio is not None and not upscale_pass_audio:
            audio_out = audio
            logging.info("[Star Minimax Latent Upscaler] passing the connected audio through to the output")
        else:
            if audio is not None:
                logging.info("[Star Minimax Latent Upscaler] 'Upscale Pass Audio' selected - "
                             "connected audio input ignored")
            if samples_in.is_nested:
                audio_src = samples if upscale_pass_audio else samples_in
                audio_out = decode_audio(audio_vae, audio_src)
            else:
                logging.info("[Star Minimax Latent Upscaler] input latent has no audio member - "
                             "returning a silent placeholder")
                audio_out = {"waveform": torch.zeros([1, 2, 4410]), "sample_rate": 44100}

        return io.NodeOutput(images, audio_out, 24.0, {"samples": samples})


class StarMinimaxLatentUpscalerOption(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StarMinimaxLatentUpscalerOption",
            display_name="⭐ Star Minimax Latent Upscaler Option",
            category="⭐StarNodes/Video",
            description=(
                "Second-pass latent upscale options for the ⭐ Star Minimax All In One "
                "node - connect to its 'options' input. The pass-1 video latent is "
                "upscaled with the selected MiniMax H3 latent upscaler model and refined "
                "in a short second sampling pass with the same conditioning (references "
                "are resolution-matched automatically) and the same seed as pass 1. The "
                "audio toggle picks which pass the audio output is decoded from."
            ),
            inputs=[
                *_settings_inputs(_upscale_model_list()),
                io.Model.Input("model", optional=True,
                               tooltip="Optional diffusion model for the refine pass (e.g. with a turbo LoRA and/or attention patch applied). If not connected, the pass-1 model is reused."),
            ],
            outputs=[
                io.Custom("UPSCALE_SETTINGS").Output("upscale_settings",
                                                     tooltip="Settings bundle for the 'options' input of ⭐ Star Minimax All In One."),
            ],
        )

    @classmethod
    def execute(cls, upscale_model, megapixels, sigmas_preset, sampler_name,
                upscale_pass_audio, align, enable_chunking, device, precision,
                model=None) -> io.NodeOutput:
        _check_model_name(upscale_model)
        return io.NodeOutput({
            "model": model,
            "upscale_model": upscale_model,
            "megapixels": float(megapixels),
            "sigmas": _parse_sigmas(sigmas_preset),
            "sigmas_preset": sigmas_preset,
            "sampler_name": sampler_name,
            "upscale_pass_audio": bool(upscale_pass_audio),
            "align": int(align),
            "enable_chunking": bool(enable_chunking),
            "device": device,
            "precision": precision,
        })


NODE_CLASS_MAPPINGS = {
    "StarMinimaxLatentUpscaler": StarMinimaxLatentUpscaler,
    "StarMinimaxLatentUpscalerOption": StarMinimaxLatentUpscalerOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxLatentUpscaler": "⭐ Star Minimax Latent Upscaler",
    "StarMinimaxLatentUpscalerOption": "⭐ Star Minimax Latent Upscaler Option",
}
