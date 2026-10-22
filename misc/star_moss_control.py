import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import folder_paths

# The MOSS color/depth control adapter replaces the model's input projection
# (128 -> 384 channels) so each denoise step takes
# [noisy latent | color map latent | depth map latent]. A zero-filled map means
# "no control for this channel". A low-rank LoRA on the transformer blocks is
# part of the same adapter file.
EXPECTED_LATENT_CHANNELS = 128

_adapter_cache = {}


def _read_adapter_file(file_name):
    """Load and validate the control adapter safetensors from models/model_patches."""
    file_path = folder_paths.get_full_path_or_raise("model_patches", file_name)
    if file_path not in _adapter_cache:
        state_dict, metadata = comfy.utils.load_torch_file(file_path, safe_load=True, return_metadata=True)
        channel_info = (metadata or {}).get("control_channels", "")
        if "video_embedder.weight" not in state_dict or channel_info != "color,depth":
            raise ValueError(
                f"{file_name} is not a MOSS color/depth control adapter "
                "(missing video_embedder weights or control_channels metadata)."
            )
        # keep only the most recently used adapter in memory
        _adapter_cache.clear()
        _adapter_cache[file_path] = state_dict
    return _adapter_cache[file_path]


class _MossEmbedderPatch(torch.nn.Module):
    """Replacement input projection for the control adapter.

    Weights are stored on CPU and moved to the active device/dtype lazily so
    ComfyUI's memory management keeps working.
    """

    def __init__(self, weight, bias):
        super().__init__()
        self._weight_cpu = weight
        self._bias_cpu = bias
        self._cast_key = None

    @property
    def weight(self):
        # MOSSImage.concat_cond reads the widened input size from here
        return self._weight_cpu

    def forward(self, x):
        key = (x.device, x.dtype)
        if self._cast_key != key:
            self._weight = self._weight_cpu.to(device=x.device, dtype=x.dtype)
            self._bias = self._bias_cpu.to(device=x.device, dtype=x.dtype)
            self._cast_key = key
        return F.linear(x, self._weight, self._bias)


def _encode_control_map(vae, image):
    """Encode a control image into FLUX.2 latents at its native size plus a
    half-size version (used when sampling starts at reduced resolution)."""
    if image is None:
        return None
    pixels = image[:1, :, :, :3]
    latent_h, latent_w = pixels.shape[1] // 16, pixels.shape[2] // 16
    target_sizes = (
        (latent_h, latent_w),
        (max(2, round(latent_h / 2)), max(2, round(latent_w / 2))),
    )
    encoded = {}
    for lh, lw in target_sizes:
        if (lh * 16, lw * 16) != tuple(pixels.shape[1:3]):
            resized = comfy.utils.common_upscale(
                pixels.movedim(-1, 1), lw * 16, lh * 16, "lanczos", "center"
            ).movedim(1, -1)
        else:
            resized = pixels
        latent = vae.encode(resized).float().cpu()
        if latent.ndim != 4 or latent.shape[1] != EXPECTED_LATENT_CHANNELS:
            kind = "video" if latent.ndim == 5 else f"{latent.shape[1]}-channel"
            raise ValueError(
                f"Star Moss Control requires the FLUX.2 VAE. The connected VAE produced a "
                f"{kind} latent instead of {EXPECTED_LATENT_CHANNELS} channels."
            )
        encoded[tuple(latent.shape[-2:])] = latent
    return encoded


def _sampling_progress(transformer_options):
    """Return how far along the sampling schedule we are (0.0 - 1.0)."""
    sigmas = transformer_options.get("sigmas")
    schedule = transformer_options.get("sample_sigmas")
    if sigmas is None or schedule is None or len(schedule) < 2:
        return 0.0
    step = int(torch.argmin((schedule.to(sigmas.device).float() - sigmas.float().max()).abs()).item())
    return step / max(len(schedule) - 1, 1)


class StarMossControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MOSS Image model (SFT or Base)."}),
                "vae": ("VAE", {"tooltip": "FLUX.2 VAE used to encode the control maps."}),
                "adapter_file": (folder_paths.get_filename_list("model_patches"),
                                 {"tooltip": "MOSS color/depth control adapter safetensors "
                                             "(models/model_patches)."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                            "tooltip": "Strength of the adapter's LoRA. The replaced "
                                                       "input layer is always applied fully."}),
                "color_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Fraction of steps after which the color map stops "
                                                   "guiding. Real photo maps: 1.0. Painted/flat maps: 0.4-0.5."}),
                "depth_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Fraction of steps after which the depth map stops "
                                                   "guiding. Real photo depth: 1.0. Rough/mannequin depth: "
                                                   "lower (or disconnect the map)."}),
            },
            "optional": {
                "color_map": ("IMAGE", {"tooltip": "Flat color region map (colors act as region IDs; "
                                                   "black = unspecified). Same size as the target image."}),
                "depth_map": ("IMAGE", {"tooltip": "Grayscale depth map, near = white. "
                                                   "Same size as the target image."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "⭐StarNodes/MOSS"
    DESCRIPTION = ("Color/depth ControlNet for MOSS Image: guides generation with a color "
                   "region map and/or a depth map. Either input can be left unconnected.")

    def apply(self, model, vae, adapter_file, lora_strength, color_end, depth_end,
              color_map=None, depth_map=None):
        if type(model.model).__name__ != "MOSSImage":
            raise ValueError("Star Moss Control only works with a MOSS Image model.")
        if color_map is None and depth_map is None:
            return (model,)

        state_dict = _read_adapter_file(adapter_file)
        patched = model.clone()

        # 1) swap the input projection for the adapter's 384-channel version
        embedder = _MossEmbedderPatch(
            state_dict["video_embedder.weight"].float(),
            state_dict["video_embedder.bias"].float(),
        )
        patched.add_object_patch("diffusion_model.video_embedder", embedder)

        # 2) apply the adapter's LoRA weights to the transformer
        if lora_strength:
            key_map = comfy.lora.model_lora_keys_unet(patched.model, {})
            lora_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items() if ".lora_" in k}
            lora_patches = comfy.lora.load_lora(lora_dict, key_map, log_missing=False)
            if not lora_patches:
                raise ValueError("The adapter's LoRA did not match any layer of this model.")
            patched.add_patches(lora_patches, lora_strength)

        # 3) encode the control maps; a missing map becomes zeros ("no control")
        color_latents = _encode_control_map(vae, color_map)
        depth_latents = _encode_control_map(vae, depth_map)
        if color_latents is None:
            color_latents = {size: torch.zeros_like(t) for size, t in depth_latents.items()}
        if depth_latents is None:
            depth_latents = {size: torch.zeros_like(t) for size, t in color_latents.items()}

        ctx = {
            "color": color_latents,
            "depth": depth_latents,
            "use_color": color_map is not None,
            "use_depth": depth_map is not None,
            "resized": {},
        }

        def _latent_for(latents, x):
            """Pick the encoded map matching x's spatial size (closest encode,
            resized if needed), on x's device/dtype and batch."""
            size = tuple(x.shape[-2:])
            cache_key = (id(latents), size, x.device, x.dtype)
            cached = ctx["resized"].get(cache_key)
            if cached is None:
                cached = latents.get(size)
                if cached is None:
                    nearest = min(
                        latents.values(),
                        key=lambda t: abs(t.shape[-2] - size[0]) + abs(t.shape[-1] - size[1]),
                    )
                    cached = F.interpolate(nearest, size=size, mode="bilinear", align_corners=False)
                cached = cached.to(device=x.device, dtype=x.dtype)
                if len(ctx["resized"]) > 8:
                    ctx["resized"].clear()
                ctx["resized"][cache_key] = cached
            return cached.expand(x.shape[0], -1, -1, -1)

        def _diffusion_wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
            progress = _sampling_progress(transformer_options)
            color = _latent_for(ctx["color"], x)
            depth = _latent_for(ctx["depth"], x)
            if not ctx["use_color"] or progress >= color_end:
                color = torch.zeros_like(color)
            if not ctx["use_depth"] or progress >= depth_end:
                depth = torch.zeros_like(depth)
            # concat_cond already appended the widened input slots; fill them
            extra = color.shape[1] + depth.shape[1]
            if x.shape[1] > EXPECTED_LATENT_CHANNELS:
                x = x[:, :x.shape[1] - extra]
            return executor(
                torch.cat((x, color, depth), dim=1), timestep, context, transformer_options, **kwargs
            )

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "star_moss_control", _diffusion_wrapper
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "StarMossControl": StarMossControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMossControl": "⭐ Star Moss Control",
}
