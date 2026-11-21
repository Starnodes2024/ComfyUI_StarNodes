"""Star Model Loader

A preset-driven model loader: pick a checkpoint (model + clip + vae), up to
three diffusion models, two text encoders and three VAEs from the dropdowns -
or apply a named preset that fills the widgets for you. The outputs are
fixed (one per slot, in the canonical order below); slots left at "(none)"
return None, so just leave their sockets unconnected. Fixed sockets keep
workflow links stable across saves and tab switches.

Presets live in json/star_model_loader_presets.json and are managed with the
Save Preset / Update Preset / Delete Preset buttons on the node.
"""

import json
import logging
import os

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

from ..image_tools.star_latent_ratio_megapixel import StarAdvanvesRatioLatent

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRESETS_PATH = os.path.join(_PACK_DIR, "json", "star_model_loader_presets.json")

_NONE = "(none)"

# widget name -> preset key + output sockets (fixed order = socket and
# tuple order; unselected slots output None)
_SLOTS = (
    ("Checkpoint", "checkpoint", (("checkpoint_model", "MODEL"),
                                  ("checkpoint_clip", "CLIP"),
                                  ("checkpoint_vae", "VAE"))),
    ("Model 1", "model_1", (("model_1", "MODEL"),)),
    ("Model 2", "model_2", (("model_2", "MODEL"),)),
    ("Model 3", "model_3", (("model_3", "MODEL"),)),
    ("Text-Encoder 1", "text_encoder_1", (("text_encoder_1", "CLIP"),)),
    ("Text-Encoder 2", "text_encoder_2", (("text_encoder_2", "CLIP"),)),
    ("VAE 1", "vae_1", (("vae_1", "VAE"),)),
    ("VAE 2", "vae_2", (("vae_2", "VAE"),)),
    ("VAE 3", "vae_3", (("vae_3", "VAE"),)),
    # sampler/scheduler outputs carry the real combo value lists, so a
    # widget converted to input (input_type = the list) validates directly
    ("Sampler", "sampler", (("sampler", comfy.samplers.KSampler.SAMPLERS),)),
    ("Scheduler", "scheduler", (("scheduler", comfy.samplers.KSampler.SCHEDULERS),)),
    ("Steps", "steps", (("steps", "INT"),)),
)
_PRESET_KEYS = tuple(key for _w, key, _o in _SLOTS)

# Advanced-only latent widgets: (widget name, preset key, default). These are
# stored in the same presets file - the basic loader ignores them, and presets
# saved before this version are migrated on load (missing keys get defaults).
_LATENT_WIDGETS = (
    ("Ratio", "latent_ratio", "1:1"),
    ("Custom Ratio", "latent_custom_ratio", "21:9"),
    ("Resolution", "latent_resolution", "SDXL (≈ 1024x1024)"),
    ("Custom Width", "latent_width", 1920),
    ("Custom Height", "latent_height", 1080),
    ("Latent Channels", "latent_channels", "128 (FLUX2)"),
    ("Ratio From Image", "latent_from_image", False),
    ("Batch Size", "latent_batch", 1),
)

_PRESET_DEFAULTS = {key: _NONE for _w, key, _o in _SLOTS}
_PRESET_DEFAULTS["steps"] = 0
_PRESET_DEFAULTS.update({key: d for _w, key, d in _LATENT_WIDGETS})
_ALL_PRESET_KEYS = tuple(_PRESET_DEFAULTS)

# filename hints -> comfy.sd.CLIPType (longer keys first); anything unknown
# falls back to STABLE_DIFFUSION, like the other StarNodes loaders
_CLIP_HINTS = (
    ("hunyuan_video_15", "HUNYUAN_VIDEO_15"), ("hunyuan_video", "HUNYUAN_VIDEO"),
    ("hunyuan_image", "HUNYUAN_IMAGE"), ("hunyuan", "HUNYUAN_DIT"),
    ("flux2", "FLUX2"), ("flux", "FLUX"), ("qwen", "QWEN_IMAGE"),
    ("wan", "WAN"), ("sd3", "SD3"), ("ltxv", "LTXV"), ("mochi", "MOCHI"),
    ("pixart", "PIXART"), ("cosmos", "COSMOS"), ("lumina", "LUMINA2"),
    ("hidream", "HIDREAM"), ("chroma", "CHROMA"), ("omnigen", "OMNIGEN2"),
    ("ace", "ACE"), ("ovis", "OVIS"), ("kandinsky", "KANDINSKY5"),
    ("newbie", "NEWBIE"),
)


def _clip_type(filename):
    low = filename.lower()
    for key, name in _CLIP_HINTS:
        if key in low:
            return getattr(comfy.sd.CLIPType, name)
    return comfy.sd.CLIPType.STABLE_DIFFUSION


def _load_presets():
    try:
        with open(_PRESETS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    # migrate entries written before later fields existed (latent_*, steps,
    # sampler, ...) - fill the missing keys once and persist
    changed = False
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        for key, default in _PRESET_DEFAULTS.items():
            if key not in entry:
                entry[key] = default
                changed = True
    if changed:
        try:
            _save_presets(data)
        except Exception:
            pass
    return data


def _save_presets(presets):
    os.makedirs(os.path.dirname(_PRESETS_PATH), exist_ok=True)
    tmp = _PRESETS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, _PRESETS_PATH)


class StarModelLoader:
    """One model picker for a whole workflow: checkpoint, diffusion models,
    text encoders and VAEs - one fixed output socket per slot, None for
    slots left at '(none)'.
    Presets are stored in json/star_model_loader_presets.json."""

    @classmethod
    def INPUT_TYPES(cls):
        presets = [_NONE] + sorted(_load_presets())
        ckpts = [_NONE] + folder_paths.get_filename_list("checkpoints")
        models = [_NONE] + folder_paths.get_filename_list("diffusion_models")
        encs = [_NONE] + folder_paths.get_filename_list("text_encoders")
        vaes = [_NONE] + folder_paths.get_filename_list("vae")
        return {
            "required": {
                "Preset": (presets, {
                    "tooltip": "Apply a saved preset to the model dropdowns "
                               "(managed with the Save/Update/Delete Preset "
                               "buttons). The run uses the widget values, "
                               "whatever set them."}),
                "Checkpoint": (ckpts, {
                    "tooltip": "Load a checkpoint - creates three outputs: "
                               "checkpoint_model (MODEL), checkpoint_clip "
                               "(CLIP) and checkpoint_vae (VAE). '(none)' "
                               "skips it."}),
                "Model 1": (models, {
                    "tooltip": "Load a diffusion model (MODEL output). "
                               "'(none)' skips it."}),
                "Model 2": (models, {
                    "tooltip": "Load a diffusion model (MODEL output). "
                               "'(none)' skips it."}),
                "Model 3": (models, {
                    "tooltip": "Load a diffusion model (MODEL output). "
                               "'(none)' skips it."}),
                "Text-Encoder 1": (encs, {
                    "tooltip": "Load a text encoder (CLIP output). The "
                               "encoder type is detected from the filename "
                               "(flux, sd3, wan, qwen, ...) with "
                               "stable_diffusion as fallback."}),
                "Text-Encoder 2": (encs, {
                    "tooltip": "Load a text encoder (CLIP output). The "
                               "encoder type is detected from the filename "
                               "(flux, sd3, wan, qwen, ...) with "
                               "stable_diffusion as fallback."}),
                "VAE 1": (vaes, {
                    "tooltip": "Load a VAE (VAE output). '(none)' skips it."}),
                "VAE 2": (vaes, {
                    "tooltip": "Load a VAE (VAE output). '(none)' skips it."}),
                "VAE 3": (vaes, {
                    "tooltip": "Load a VAE (VAE output). '(none)' skips it."}),
                "Sampler": ([_NONE] + comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Pass the sampler name through (COMBO output - "
                               "wire it to a sampler_name widget converted to "
                               "input). '(none)' skips it."}),
                "Scheduler": ([_NONE] + comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Pass the scheduler name through (COMBO "
                               "output - wire it to a scheduler widget "
                               "converted to input). '(none)' skips it."}),
                "Steps": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Pass a step count through (INT output - "
                               "wire it to a steps input/widget converted "
                               "to input). 0 = no steps output."}),
            },
        }

    # fixed sockets, one per slot - real types, so the declared definition
    # always matches the visible node and restored links keep their slots
    RETURN_TYPES = tuple(t for _w, _k, outs in _SLOTS for _n, t in outs)
    RETURN_NAMES = tuple(name for _w, _k, outs in _SLOTS for name, _t in outs)
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = ("Load a checkpoint, diffusion models, text encoders and "
                   "VAEs from one node - fixed outputs, None for unselected "
                   "slots. Save named presets of the selections "
                   "(json/star_model_loader_presets.json).")

    def _load_slots(self, kw):
        """Run the model-slot selections; returns one value per fixed output
        socket in slot order (None for unselected slots)."""
        outs = []
        for wname, key, slot_outs in _SLOTS:
            name = kw.get(wname)
            if not name or name == _NONE:
                outs += [None] * len(slot_outs)
                continue
            if key == "checkpoint":
                path = folder_paths.get_full_path_or_raise("checkpoints", name)
                model, clip, vae, _cv = comfy.sd.load_checkpoint_guess_config(
                    path, output_vae=True, output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths(
                        "embeddings"))
                outs += [model, clip, vae]
            elif key.startswith("model_"):
                path = folder_paths.get_full_path_or_raise("diffusion_models", name)
                outs.append(comfy.sd.load_diffusion_model(path))
            elif key.startswith("text_encoder"):
                path = folder_paths.get_full_path_or_raise("text_encoders", name)
                outs.append(comfy.sd.load_clip(
                    ckpt_paths=[path],
                    embedding_directory=folder_paths.get_folder_paths(
                        "embeddings"),
                    clip_type=_clip_type(name)))
            elif key in ("sampler", "scheduler"):
                outs.append(name)   # COMBO output: the name string
            elif key == "steps":
                outs.append(int(name))
            else:
                path = folder_paths.get_full_path_or_raise("vae", name)
                outs.append(comfy.sd.VAE(
                    sd=comfy.utils.load_torch_file(path, safe_load=True)))
            logging.info("[Star Model Loader] %s -> %s", wname, name)
        return outs

    def load(self, **kw):
        outs = self._load_slots(kw)
        if all(o is None for o in outs):
            raise ValueError("[Star Model Loader] nothing selected - pick a "
                             "checkpoint, model, text encoder, VAE, sampler, "
                             "scheduler or steps, or apply a preset.")
        return tuple(outs)


# option lists shared with the Star Advanced Ratio/Latent node - calling its
# INPUT_TYPES also fills its _RATIO_MAP, which create() needs
_LATENT_SRC = StarAdvanvesRatioLatent.INPUT_TYPES()["required"]


class StarModelLoaderAdvanced(StarModelLoader):
    """Star Model Loader plus the Star Advanced Ratio/Latent generator: same
    slots and the same shared presets file (latent settings are stored as
    extra keys the basic loader ignores), plus a latent/width/height output."""

    @classmethod
    def INPUT_TYPES(cls):
        required = dict(StarModelLoader.INPUT_TYPES()["required"])
        required.update({
            "Ratio": (_LATENT_SRC["ratio"][0], {
                "default": "1:1",
                "tooltip": "Aspect ratio for the generated latent. 'custom' "
                           "uses the Custom Ratio field."}),
            "Custom Ratio": ("STRING", {
                "default": "21:9", "multiline": False,
                "placeholder": "e.g. 21:9",
                "tooltip": "Used when Ratio is 'custom' (e.g. '21:9' or "
                           "'21x9')."}),
            "Resolution": (_LATENT_SRC["resolution"][0], {
                "default": "SDXL (≈ 1024x1024)",
                "tooltip": "Target size for the generated latent - a "
                           "megapixel budget or 'custom' for the "
                           "Width/Height fields."}),
            "Custom Width": ("INT", {
                "default": 1920, "min": 16, "max": 99968, "step": 16,
                "tooltip": "Used when Resolution is 'custom'."}),
            "Custom Height": ("INT", {
                "default": 1080, "min": 16, "max": 99968, "step": 16,
                "tooltip": "Used when Resolution is 'custom'."}),
            "Latent Channels": (_LATENT_SRC["latent_channels"][0], {
                "default": "128 (FLUX2)",
                "tooltip": "Channel count of the generated latent - pick "
                           "the one matching your model."}),
            "Ratio From Image": ("BOOLEAN", {
                "default": False,
                "tooltip": "Use the aspect ratio of the connected image "
                           "(nearest preset ratio) instead of the Ratio "
                           "selection."}),
            "Batch Size": ("INT", {
                "default": 1, "min": 1, "max": 4096,
                "tooltip": "Number of latents in the batch."}),
        })
        return {
            "required": required,
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image - when 'Ratio From Image' is "
                               "on, the latent gets its aspect ratio "
                               "(nearest preset ratio)."}),
            },
        }

    # fixed slot sockets + always-on latent/width/height at the end
    RETURN_TYPES = StarModelLoader.RETURN_TYPES + ("LATENT", "INT", "INT")
    RETURN_NAMES = StarModelLoader.RETURN_NAMES + ("latent", "width", "height")
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = ("Star Model Loader plus a Star Advanced Ratio/Latent "
                   "generator - same fixed model outputs, plus a "
                   "latent/width/height output. Shares the presets file "
                   "with the basic loader.")

    def load(self, **kw):
        outs = self._load_slots(kw)
        outs.extend(StarAdvanvesRatioLatent.create(
            ratio=kw.get("Ratio", "1:1"),
            custom_ratio=kw.get("Custom Ratio", "21:9"),
            resolution=kw.get("Resolution", "SDXL (≈ 1024x1024)"),
            custom_width=kw.get("Custom Width", 1920),
            custom_height=kw.get("Custom Height", 1080),
            latent_channels=kw.get("Latent Channels", "128 (FLUX2)"),
            ratio_from_image=kw.get("Ratio From Image", False),
            image=kw.get("image"),
            batch_size=kw.get("Batch Size", 1)))
        logging.info("[Star Model Loader] latent %sx%s (batch %s)",
                     outs[-2], outs[-1], kw.get("Batch Size", 1))
        return tuple(outs)


NODE_CLASS_MAPPINGS = {
    "StarModelLoader": StarModelLoader,
    "StarModelLoaderAdvanced": StarModelLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarModelLoader": "⭐ Star Model Loader",
    "StarModelLoaderAdvanced": "⭐ Star Model Loader Advanced",
}


# ---------------------------------------------------------------------------
# preset endpoints (json/star_model_loader_presets.json)
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web as _web

    @PromptServer.instance.routes.get("/starnodes/model_loader/presets")
    async def _sl_presets(request):
        return _web.json_response({"status": "ok", "presets": _load_presets()})

    @PromptServer.instance.routes.post("/starnodes/model_loader/presets")
    async def _sl_preset_edit(request):
        try:
            body = await request.json()
            action = body.get("action")
            name = (body.get("name") or "").strip()
            if not name or len(name) > 80:
                return _web.json_response(
                    {"status": "error", "message": "give a preset name"}, status=400)
            presets = _load_presets()
            if action == "save":
                if name in presets:
                    return _web.json_response(
                        {"status": "error",
                         "message": f"preset '{name}' exists - use Update instead"},
                        status=409)
            elif action == "update":
                if name not in presets:
                    return _web.json_response(
                        {"status": "error",
                         "message": f"preset '{name}' not found"}, status=404)
            if action in ("save", "update"):
                # merge over the old entry: keys the frontend did not send
                # (e.g. latent_* from a basic-loader save) keep their stored
                # value; never-stored keys get the defaults
                values = body.get("values") or {}
                old = presets.get(name, {})
                presets[name] = {k: values.get(k, old.get(k, _PRESET_DEFAULTS[k]))
                                 for k in _ALL_PRESET_KEYS}
            elif action == "delete":
                if name not in presets:
                    return _web.json_response(
                        {"status": "error",
                         "message": f"preset '{name}' not found"}, status=404)
                del presets[name]
            else:
                return _web.json_response(
                    {"status": "error", "message": "unknown action"}, status=400)
            _save_presets(presets)
            return _web.json_response({"status": "ok", "presets": presets})
        except Exception as e:
            return _web.json_response(
                {"status": "error", "message": str(e)}, status=500)
except Exception:
    pass
