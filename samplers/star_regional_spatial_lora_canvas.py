"""Star Regional Spatial LoRA Canvas - per-region prompt + LoRA.

The companion JS widget (web/js/star_regional_spatial_lora_canvas.js) provides
an interactive canvas where bounding boxes are drawn; every box gets its own
short prompt, optional LoRA and LoRA strength. Boxes are serialized into the
hidden ``regions_json`` widget as normalized (0..1) coordinates:

    [{"id": 1, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.5,
      "prompt": "a young cyberpunk man with blue hair",
      "lora_name": "cyberpunk.safetensors", "lora_strength": 0.8,
      "face_detail": false}, ...]

How a region is rendered:

- Every region is one extra full-canvas model evaluation per step with its own
  prompt and its own LoRA. The predictions are merged with region weights that
  sum to exactly 1.0 on every latent pixel, so nothing gets averaged away.
- LoRAs never touch the model weights. They run as a low-rank side path in the
  forward pass of the patched layers and are switched on only for the
  evaluation of their own region. That works on quantized checkpoints
  (int8 / nvfp4 / fp8) without requantizing and without weight swapping.
- The negative side mirrors the regions, so with CFG > 1 the guidance compares
  the same LoRA on both sides.
"""

import json
import logging
import os
import uuid
import weakref

import torch

import folder_paths
import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import comfy.weight_adapter
from nodes import CLIPTextEncode

from ..image_tools.star_latent_ratio_megapixel import StarAdvanvesRatioLatent

# --- embedded refiner system prompts ---------------------------------------
# Region prompts: describe ONLY the subject in the box, grounded to its canvas
# position, style-locked to the global scene. Trigger words / LoRA tokens must
# survive verbatim.
_REGION_SYSTEM_PROMPT = """You are a prompt writer for masked regional text-to-image generation. The canvas is split into regions; every region gets its own subject prompt, while a global scene prompt styles the whole picture.

Task: rewrite the given region prompt for ONE bounded region of the image.

Rules:
- Keep style, lighting, mood and color palette consistent with the global scene. Never introduce a different style or contradict the scene.
- Describe ONLY the subject inside this region: appearance, clothing/fur/surface, expression, pose, material, key identifying details. No full background, no other characters, no framing of the whole scene.
- Ground the position naturally from the given normalized box coordinates ("on the left", "center-right", "in the foreground", ...).
- Preserve ALL existing trigger words and tags exactly as written (LoRA trigger words, character names, style tags). Never reformat, merge or translate them.
- Write one dense, flowing description of 1-3 sentences. No keyword chains, no lists, no in quotes text, no commentary, no negative words like "no" or "without".
- If the region prompt is empty or unusable, keep it short and generic for the box position.

Output ONLY the final region prompt text."""

# Global scene prompt: composition/style/context only, no per-subject detail.
_GLOBAL_SYSTEM_PROMPT = """You are a scene prompt writer for masked multi-region text-to-image generation. The user gives a rough description of one image composed of several regions; every region has its own subject prompt that is applied separately.

Task: rewrite the input into the GLOBAL background/context prompt.

The global prompt must carry: overall art style and quality intent, composition and camera angle, lighting setup, environment and atmosphere, color palette. It may reference the regions' subjects only generically (e.g. "group of people on the left and right") - never any per-subject detail.

Rules:
- Keep every style/quality/technical intent of the input text.
- Do not add subjects or details that are not in the input.
- One dense, flowing paragraph. No lists, no quotes, no commentary, no negative words like "no" or "without".

Output ONLY the final global prompt."""


def _clip_generate(clip, user_text, system_prompt, seed, max_length=512):
    """Small instruct-LLM generation via a text encoder (like ⭐ Star CLIP Prompt Refiner)."""
    tokens = clip.tokenize(user_text, min_length=1, thinking=True, system_prompt=system_prompt)
    generated_ids = clip.generate(
        tokens,
        do_sample=True,
        max_length=max_length,
        temperature=0.7,
        top_k=64,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.05,
        seed=seed,
    )
    generated_text = clip.decode(generated_ids)
    reasoning, _, text = generated_text.partition("</think>")
    if not reasoning.lstrip().startswith("<think>"):
        text = generated_text
    return text.strip()



# --- regional LoRA side path -------------------------------------------------
_REGION_KEY = "star_regional_lora"
_DEBUG = os.environ.get("STAR_REGIONAL_DEBUG") == "1"

# One forward wrapper per module per underlying model, shared across clones and
# canvas builds: a rebuilt canvas must never stack another wrapper on top.
# Adapters stay separated by build-unique token ({build_id}:{index}).
_LAYERS = weakref.WeakKeyDictionary()

# injection key -> runtime counters; StarRegionalSampler reads these back to
# warn when a registered LoRA never fired instead of failing silently.
_REGION_STATS = {}
_MAX_ADAPTER_TOKENS = 48


class _RegionSwitch:
    """The region token for the model call that is running."""
    active = None


class _RegionLoraLayer:
    """Runs one patched layer as base + the LoRA side path of the active region."""

    def __init__(self, module):
        self.module = module
        self.adapters = {}
        self.original_forward = None

    def add(self, token, adapter, strength, out_slice=None):
        module = self.module
        conv_dim = next((d for d, c in ((1, torch.nn.Conv1d), (2, torch.nn.Conv2d), (3, torch.nn.Conv3d))
                         if isinstance(module, c)), 0)
        adapter.multiplier = strength
        adapter.is_conv = conv_dim > 0
        adapter.conv_dim = conv_dim
        adapter.kernel_size = module.kernel_size if conv_dim else (1,)
        adapter.in_channels = module.in_channels if conv_dim else None
        adapter.out_channels = module.out_channels if conv_dim else None
        adapter.kw_dict = {"stride": module.stride, "padding": module.padding,
                           "dilation": module.dilation, "groups": module.groups} if conv_dim else {}
        self.adapters.setdefault(token, []).append((adapter, out_slice))
        while len(self.adapters) > _MAX_ADAPTER_TOKENS:
            self.adapters.pop(next(iter(self.adapters)))  # evict dead builds

    def forward(self, x, *args, **kwargs):
        if self.original_forward is None:
            return x  # ejected while another wrapper still references us
        token = _RegionSwitch.active
        entries = self.adapters.get(token, ())
        if not entries:
            return self.original_forward(x, *args, **kwargs)
        stats = _REGION_STATS.get(f"{_REGION_KEY}_{token.partition(':')[0]}")
        if stats is not None:
            stats["adapter_calls"] += 1
        for adapter, _out_slice in entries:
            _follow_input(adapter, x)
        forward = self.original_forward
        for adapter, out_slice in entries:
            if out_slice is None:
                forward = _bind_adapter(adapter, forward)
        out = forward(x, *args, **kwargs)
        for adapter, out_slice in entries:
            if out_slice is None:
                continue
            # adapter for a slice of a fused weight (e.g. to_q inside qkv):
            # its delta lands on output features [start, start+length)
            _dim, start, length = out_slice
            out_dim = 1 if adapter.is_conv else -1
            target = out.narrow(out_dim, start, length)
            delta = adapter.h(x, out)
            if delta.shape == target.shape:
                target.copy_(adapter.g(target + delta))
        return out

    def inject(self):
        if self.original_forward is not None:
            return
        device = comfy.model_management.get_torch_device()
        weight = getattr(self.module, "weight", None)
        dtype = weight.dtype if isinstance(weight, torch.Tensor) else None
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = None  # quantized layer: the side path follows the activation dtype
        for adapters in self.adapters.values():
            for adapter, _out_slice in adapters:
                adapter._rdk = None
                adapter.weights = tuple(w.to(device=device, dtype=dtype) if isinstance(w, torch.Tensor) else w
                                        for w in adapter.weights)
        self.original_forward = self.module.forward
        self.original_force_forward = getattr(self.module, "comfy_force_forward", False)
        # fused fast paths skip module.forward unless the layer asks for it
        self.module.comfy_force_forward = True
        self.module.forward = self.forward

    def eject(self):
        if self.original_forward is None:
            return
        if self.module.forward == self.forward:  # restore only while still top of the chain
            self.module.forward = self.original_forward
            self.module.comfy_force_forward = self.original_force_forward
        self.original_forward = None


def _follow_input(adapter, x):
    """Recast adapter weights when the activation dtype/device changed (cached)."""
    key = (x.device, x.dtype)
    if getattr(adapter, "_rdk", None) != key:
        adapter.weights = tuple(w.to(device=x.device, dtype=x.dtype)
                                if isinstance(w, torch.Tensor) and w.is_floating_point() else w
                                for w in adapter.weights)
        adapter._rdk = key


def _bind_adapter(adapter, forward):
    return lambda x, *args, **kwargs: adapter.bypass_forward(forward, x, *args, **kwargs)


def _attach_region_lora(model, token, lora, strength):
    """Registers one region's LoRA on the layers it targets. Returns the entry count."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map, log_missing=False)
    layers = _LAYERS.setdefault(model.model, {})
    attached = 0
    skipped = 0
    for key, adapter in loaded.items():
        if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            # plain weight diffs / set entries have no forward-pass form
            skipped += 1
            continue
        out_slice = None
        if isinstance(key, tuple):
            # (weight_key, (dim, start, length)): adapter on a slice of a fused
            # weight, e.g. diffusers to_q/to_k/to_v inside a fused qkv linear
            if len(key) != 2 or not isinstance(key[0], str) or key[1] is None or key[1][0] != 0:
                skipped += 1
                continue
            key, out_slice = key
        if not isinstance(key, str) or not key.endswith(".weight"):
            skipped += 1
            continue
        module_key = key[:-len(".weight")]
        if module_key not in layers:
            layers[module_key] = _RegionLoraLayer(comfy.utils.get_attr(model.model, module_key))
        layers[module_key].add(token, adapter, strength, out_slice=out_slice)
        attached += 1
    if skipped:
        logging.warning(f"[Star Regional LoRA Canvas] {skipped} LoRA entries are not regional-capable (weight diffs / transformed weights) and were skipped")
    return attached


def _install_region_loras(model, layers, build_id):
    stats = _REGION_STATS[f"{_REGION_KEY}_{build_id}"] = {
        "inject_runs": 0, "model_calls": 0, "tokens_seen": set(), "adapter_calls": 0}
    while len(_REGION_STATS) > 32:
        _REGION_STATS.pop(next(iter(_REGION_STATS)))

    def inject(patcher):
        stats["inject_runs"] += 1
        if _DEBUG:
            logging.info(f"[Star Regional LoRA Canvas] injecting {len(layers)} region LoRA layers")
        for layer in layers.values():
            layer.inject()

    def eject(patcher):
        for layer in layers.values():
            layer.eject()

    def apply_model(executor, *args, **kwargs):
        stats["model_calls"] += 1
        to = kwargs.get("transformer_options")
        if not isinstance(to, dict) or _REGION_KEY not in to:
            to = next((a for a in args if isinstance(a, dict) and _REGION_KEY in a),
                      args[5] if len(args) > 5 and isinstance(args[5], dict) else {})
        token = to.get(_REGION_KEY)
        if token is not None:
            stats["tokens_seen"].add(token)
        if _DEBUG:
            logging.info(f"[Star Regional LoRA Canvas] model eval, region token={token}")
        _RegionSwitch.active = token
        try:
            return executor(*args, **kwargs)
        finally:
            _RegionSwitch.active = None

    # unique key: a rebuilt region set must never be mistaken for the loaded one
    model.set_injections(f"{_REGION_KEY}_{build_id}", [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, _REGION_KEY, apply_model)
    inject(model)  # apply now; the patcher injection re-applies after every unload


def _region_hooks(token):
    """Marks a conditioning entry so the sampler evaluates it on its own with its region token."""
    hooks = comfy.hooks.HookGroup()
    hooks.add(comfy.hooks.TransformerOptionsHook({_REGION_KEY: token}, hook_scope=comfy.hooks.EnumHookScope.HookedOnly))
    return hooks


# --- region geometry ---------------------------------------------------------
def _axis_ramp(size, start, end, feather):
    pos = torch.arange(size, dtype=torch.float32) + 0.5
    if feather <= 0.0:
        return ((pos >= start) & (pos < end)).float()
    far = torch.full_like(pos, feather)
    # a side that sits on the canvas border stays hard, the image edge is no seam
    lo = (pos - start) if start > 0.5 else torch.where(pos >= start, far, pos - start)
    hi = (end - pos) if end < size - 0.5 else torch.where(pos < end, far, end - pos)
    return (torch.minimum(lo, hi) / feather).clamp(0.0, 1.0)


def _build_region_mask(w_latent, h_latent, region, feather):
    """Normalized box (x, y, w, h in 0..1) -> soft mask [h_latent, w_latent]."""
    rx = max(0.0, min(1.0, float(region.get("x", 0.0))))
    ry = max(0.0, min(1.0, float(region.get("y", 0.0))))
    rw = max(0.0, min(1.0 - rx, float(region.get("w", 0.05))))
    rh = max(0.0, min(1.0 - ry, float(region.get("h", 0.05))))

    ramp_x = _axis_ramp(w_latent, rx * w_latent, (rx + rw) * w_latent, feather)
    ramp_y = _axis_ramp(h_latent, ry * h_latent, (ry + rh) * h_latent, feather)
    ramp = torch.minimum(ramp_x.unsqueeze(0), ramp_y.unsqueeze(1))
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _position_phrase(region):
    w, h = float(region.get("w", 0.0)), float(region.get("h", 0.0))
    if w > 0.8 and h > 0.8:
        return ""
    cx = float(region.get("x", 0.0)) + w / 2
    cy = float(region.get("y", 0.0)) + h / 2
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    vertical = "upper" if cy < 0.33 and h < 0.6 else "lower" if cy > 0.67 and h < 0.6 else ""
    if vertical:
        return f"in the {vertical} {horizontal} of the image"
    if horizontal == "center":
        return "in the center of the image"
    return f"on the {horizontal} side of the image"


# --- per-model starting points ------------------------------------------------
# latent_channels must match the model or the latent and the masks are wrong;
# feather is the recommended soft edge in latent pixels (used when mask_feather
# is left at the default 2.0); sampler is the recommended sampler recipe for
# the ⭐ Star Regional Sampler, logged once per run.
_MODEL_PRESETS = {
    "MOSS Image": {
        "latent_channels": "128 (FLUX2)", "feather": 2.0,
        "sampler": "sampler: ~30 steps, cfg 4.5 - a negative prompt is strongly recommended",
    },
    "Krea2": {
        "latent_channels": "16 (FLUX/QWEN/ZIT)", "feather": 4.0,
        "sampler": "sampler: ~8 steps, cfg 1.0",
    },
    "Qwen Image 2.1": {
        "latent_channels": "64 (QWEN2.1)", "feather": 2.0,
        "sampler": "sampler: ~10 steps, cfg 1.0",
    },
}


def _size_inputs():
    """Widget set of ⭐ Star Advanced Ratio/Latent, but with the 128-channel
    FLUX2 latent format as the default."""
    inputs = dict(StarAdvanvesRatioLatent.INPUT_TYPES()["required"])
    options, meta = inputs["latent_channels"]
    inputs["latent_channels"] = (options, {**meta, "default": "128 (FLUX2)"})
    return inputs


class StarRegionalSpatialLoraCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        text_encoder_files = folder_paths.get_filename_list("text_encoders")
        refiner_choices = ["none"] + text_encoder_files
        refiner_default = refiner_choices[1] if len(refiner_choices) > 1 else "none"
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                **_size_inputs(),
                "main_prompt": ("STRING", {
                    "default": "A cinematic group shot of...",
                    "multiline": True,
                }),
                "refine_prompts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "On: rewrite prompts with an instruct text encoder (like ⭐ Star CLIP Prompt Refiner) using embedded system prompts tuned for regional generation before conditioning.",
                }),
                "refine_scope": (["regions only", "main + regions"], {
                    "default": "regions only",
                    "tooltip": "Which prompts the refiner rewrites.",
                }),
                "refiner_clip": (refiner_choices, {
                    "default": refiner_default,
                    "tooltip": "Instruct text encoder from models/text_encoders used for the rewrite (e.g. Qwen3-VL). Loaded only when refine_prompts is on.",
                }),
                "refiner_seed": ("INT", {
                    "default": 0, "min": 0, "max": 2 ** 31 - 1, "control_after_generate": True,
                    "tooltip": "Refiner sampling seed; change to re-roll a rewrite.",
                }),
                "mask_feather": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1,
                    "tooltip": "Soft edge of the regions in latent pixels (0 = hard edge). Box sides on the canvas border are never feathered.",
                }),
                "regions_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                }),
                "region_prompt_mode": (["scene-aware", "region only"], {
                    "default": "scene-aware",
                    "tooltip": "scene-aware: every region is rendered with its own prompt, a position hint and the main prompt, so it stays part of the same picture. region only: the box prompt exactly as written.",
                }),
                "global_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Share of the main prompt (without LoRA) that stays inside the boxes. 0 = regions fully own their box (strongest LoRA, sharpest). Raise slightly if a region drifts away from the scene.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Only used with CFG > 1. Applied to the whole canvas and mirrored into every region with that region's LoRA.",
                }),
                "model_preset": (["auto", "MOSS Image", "Krea2", "Qwen Image 2.1"], {
                    "default": "auto",
                    "tooltip": "Recommended starting settings per model. Sets the correct latent format (64ch for Qwen Image 2.1, 128ch for MOSS, 16ch for Krea2) and the recommended mask feather (mask_feather = 2.0 uses the preset's value). 'auto' uses the widgets as set.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional: take the aspect ratio from this image (with 'Ratio From Image' on)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "MASK", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "region_masks", "latent")
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def build(self, model, clip, ratio, custom_ratio, resolution, custom_width, custom_height,
              latent_channels, ratio_from_image, batch_size, main_prompt, refine_prompts,
              refine_scope, refiner_clip, refiner_seed, mask_feather,
              regions_json, region_prompt_mode="scene-aware", global_blend=0.0,
              negative_prompt="", model_preset="auto", image=None):
        try:
            regions = json.loads(regions_json)
            if not isinstance(regions, list):
                raise ValueError("regions_json must be a JSON array")
        except (ValueError, TypeError) as e:
            logging.warning(f"[Star Regional LoRA Canvas] Invalid regions_json, ignoring boxes: {e}")
            regions = []

        refiner = None
        if refine_prompts and str(refiner_clip).lower() != "none":
            try:
                refiner_path = folder_paths.get_full_path_or_raise("text_encoders", refiner_clip)
                refiner = comfy.sd.load_clip(
                    ckpt_paths=[refiner_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
            except Exception as e:
                logging.warning(f"[Star Regional LoRA Canvas] Refiner '{refiner_clip}' failed to load ({e}) - prompts stay as written")
                refiner = None

        def refine(text, index, is_main=False):
            if refiner is None:
                return text
            position = ""
            if not is_main and index < len(regions) and isinstance(regions[index], dict):
                r = regions[index]
                position = (f"\nRegion box (normalized x, y, w, h): "
                            f"({float(r.get('x', 0)):.2f}, {float(r.get('y', 0)):.2f}, "
                            f"{float(r.get('w', 0)):.2f}, {float(r.get('h', 0)):.2f})")
            try:
                refined = _clip_generate(
                    refiner,
                    f"Global scene: {main_prompt}{position}\nRegion prompt: {text}" if not is_main else f"Scene description: {text}",
                    _GLOBAL_SYSTEM_PROMPT if is_main else _REGION_SYSTEM_PROMPT,
                    seed=int(refiner_seed) + (0 if is_main else index + 1),
                )
            except Exception as e:
                logging.warning(f"[Star Regional LoRA Canvas] Refiner failed ({e}) - using prompt as written")
                return text
            if not refined:
                return text
            logging.info(f"[Star Regional LoRA Canvas] refined {'main' if is_main else f'box {index + 1}'}: '{text[:40]}' -> '{refined[:60]}'")
            return refined

        preset = _MODEL_PRESETS.get(model_preset)
        if preset is not None:
            if latent_channels != preset["latent_channels"]:
                logging.warning(
                    f"[Star Regional LoRA Canvas] {model_preset} needs '{preset['latent_channels']}' "
                    f"latents - ignoring the latent_channels widget")
                latent_channels = preset["latent_channels"]
            if mask_feather == 2.0:
                mask_feather = preset["feather"]
            logging.info(f"[Star Regional LoRA Canvas] {model_preset} preset - recommended sampler: {preset['sampler']}")

        if refiner is not None and refine_scope == "main + regions":
            main_prompt = refine(main_prompt, -1, is_main=True)
        main_prompt = main_prompt.strip()

        # Same sizing math as ⭐ Star Advanced Ratio/Latent (shared code path)
        (latent_dict, width, height) = StarAdvanvesRatioLatent.create(
            ratio=ratio, custom_ratio=custom_ratio, resolution=resolution,
            custom_width=custom_width, custom_height=custom_height,
            latent_channels=latent_channels, ratio_from_image=ratio_from_image,
            image=image, batch_size=batch_size,
        )
        latent = latent_dict["samples"]
        h_latent, w_latent, channels = int(latent.shape[2]), int(latent.shape[3]), int(latent.shape[1])

        encoder = CLIPTextEncode()
        encoded = {}

        def encode(text):
            if text not in encoded:
                encoded[text] = encoder.encode(clip, text)[0]
            return encoded[text]

        # With dynamic VRAM, hooked conditioning makes the guider swap to a
        # non-dynamic delegate rebuilt from cached_patcher_init (samplers.py
        # inner_set_conds) - a different model instance our forward wrappers
        # would never be attached to. Clone non-dynamic up front when LoRAs are
        # used; ComfyUI drops dynamic VRAM for hooked conds anyway.
        needs_lora = any(isinstance(r, dict) and str(r.get("lora_name", "") or "").strip().lower() not in ("", "none")
                         for r in regions)
        model = model.clone(disable_dynamic=needs_lora)
        lora_files = {}
        build_id = uuid.uuid4().hex
        has_lora = False
        active = []  # (prompt, mask, hooks or None)
        face_regions = []

        for index, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            lora_name = str(region.get("lora_name", "") or "").strip()
            if lora_name.lower() == "none":
                lora_name = ""
            prompt = str(region.get("prompt", "")).strip()
            if not prompt and not lora_name:
                logging.warning(f"[Star Regional LoRA Canvas] Box {index + 1} has neither prompt nor LoRA - skipped")
                continue

            mask = _build_region_mask(w_latent, h_latent, region, mask_feather)
            if float(mask.max()) <= 0.0:
                logging.warning(f"[Star Regional LoRA Canvas] Box {index + 1} lies outside the canvas - skipped")
                continue

            if prompt:
                prompt = refine(prompt, index)
                if region_prompt_mode == "scene-aware":
                    prompt = ", ".join(p for p in (prompt.rstrip(" .,"), _position_phrase(region)) if p)
                    if main_prompt:
                        prompt = f"{prompt}. {main_prompt}"
            else:
                prompt = main_prompt

            hooks = None
            token = None
            attached = 0
            if lora_name:
                if lora_name not in lora_files:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                    lora_files[lora_name] = comfy.utils.load_torch_file(lora_path, safe_load=True)
                strength = float(region.get("lora_strength", 1.0))
                token = f"{build_id}:{index}"
                attached = _attach_region_lora(model, token, lora_files[lora_name], strength)
                if attached:
                    has_lora = True
                    hooks = _region_hooks(token)
                    logging.info(f"[Star Regional LoRA Canvas] Box {index + 1}: '{prompt[:40]}' + LoRA '{lora_name}' @ {strength} on {attached} layers")
                else:
                    logging.warning(f"[Star Regional LoRA Canvas] Box {index + 1}: LoRA '{lora_name}' matches no layer of this model - prompt-only region")
            else:
                logging.info(f"[Star Regional LoRA Canvas] Box {index + 1}: '{prompt[:40]}' (prompt-only)")

            active.append((prompt, mask, hooks))
            if region.get("face_detail", False):
                face_regions.append({
                    "x": float(region.get("x", 0)), "y": float(region.get("y", 0)),
                    "w": float(region.get("w", 0)), "h": float(region.get("h", 0)),
                    "prompt": prompt, "token": token if attached else None,
                })

        if has_lora:
            _install_region_loras(model, _LAYERS.get(model.model, {}), build_id)

        # region weights + global weight sum to exactly 1.0 on every latent pixel
        if active:
            masks = torch.stack([mask for _, mask, _ in active])
            weights = masks / masks.sum(dim=0).clamp(min=1.0) * (1.0 - global_blend)
            global_weight = 1.0 - weights.sum(dim=0)
        else:
            weights = torch.zeros((1, h_latent, w_latent), dtype=torch.float32)
            global_weight = torch.ones((h_latent, w_latent), dtype=torch.float32)

        def assemble(global_text, region_texts):
            out = []
            if float(global_weight.max()) > 1e-4:
                out += comfy.hooks.set_mask_for_conditioning(encode(global_text), global_weight, "default", 1.0)
            for text, weight, (_, _, hooks) in zip(region_texts, weights, active):
                cond = comfy.hooks.set_hooks_for_conditioning(encode(text), hooks, append_hooks=False)
                out += comfy.hooks.set_mask_for_conditioning(cond, weight, "default", 1.0)
            return out

        positive = assemble(main_prompt, [prompt for prompt, _, _ in active])
        negative = assemble(negative_prompt, [negative_prompt] * len(active))

        if face_regions:
            latent_dict["star_face_regions"] = face_regions
            latent_dict["star_face_negative"] = negative_prompt

        logging.info(
            f"[Star Regional LoRA Canvas] {len(active)} active region(s) "
            f"({sum(hooks is not None for _, _, hooks in active)} with LoRA) at {width}x{height} (latent {channels}ch {w_latent}x{h_latent})"
        )

        return (model, positive, negative, weights, latent_dict)


NODE_CLASS_MAPPINGS = {
    "StarRegionalSpatialLoraCanvas": StarRegionalSpatialLoraCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRegionalSpatialLoraCanvas": "⭐ Star Regional Spatial LoRA Canvas",
}
