"""Star Regional Sampler - two-pass sampler tuned for regional conditioning.

Pass 1 builds the picture from the regional conditioning. Pass 2 re-noises the
result a little and samples it again with the same regions and LoRAs: by then
all regions agree on one picture, so the pass only resolves soft transitions
and fine detail. An optional latent upscale between the passes adds resolution.

Options: a SIGMAS input accepts a custom sigma schedule (like
SamplerCustomAdvanced) and a STAR_PREVIEW input connects the ⭐ Star Preview
node from the comfyui_starnodes pack for a live sampling preview.
"""

import importlib.util
import logging
import os
import sys

import torch

import comfy.samplers
import comfy.utils
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
from nodes import common_ksampler

from .star_face_detailer import DEFAULT_DETECTOR, _detector_choices, detail_regions
from .star_regional_spatial_lora_canvas import _REGION_KEY, _REGION_STATS


def _check_region_lora_ran(model):
    """Warns when regional LoRAs were registered but never fired this run."""
    for key in getattr(model, "injections", {}):
        if not key.startswith(f"{_REGION_KEY}_"):
            continue
        stats = _REGION_STATS.get(key)
        if stats is None or stats["adapter_calls"] > 0:
            continue
        if stats["model_calls"] == 0:
            reason = "the canvas model output did not reach the model"
        elif not stats["tokens_seen"]:
            reason = "the region token never reached the model - the conditioning lost its hooks on the way"
        else:
            reason = "the patched layers were never called - the model bypasses module.forward"
        logging.warning(f"[Star Regional Sampler] regional LoRAs were registered but never fired - {reason}")


def _find_apply_star_preview():
    # ComfyUI registers node packs in sys.modules under their folder path, not
    # the package name, so "comfyui_starnodes.misc.star_preview" cannot be
    # imported directly. The pack imports it on load through star_sampler, so
    # find the already-loaded module.
    for name, module in sys.modules.items():
        if "comfyui_starnodes" in name and name.endswith("misc.star_preview"):
            return getattr(module, "apply_star_preview", None)
    # Pack present but the module was never imported - load it by file path.
    preview_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "comfyui_starnodes", "misc", "star_preview.py")
    if os.path.isfile(preview_file):
        spec = importlib.util.spec_from_file_location("starnodes_star_preview", preview_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "apply_star_preview", None)
    return None


apply_star_preview = _find_apply_star_preview()



def _sample_with_sigmas(model, positive, negative, cfg, seed, sampler_name, sigmas, latent):
    """Same machinery as SamplerCustomAdvanced: noise + CFGGuider + sampler + custom sigmas."""
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    noise = Noise_RandomNoise(seed)
    sampler = comfy.samplers.sampler_object(sampler_name)
    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]


class StarRegionalSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model output of ⭐ Star Regional Spatial LoRA Canvas (carries the regional LoRAs)."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Steps of the main pass. Regional generation likes a few more steps than a plain render: 8 for Krea2, 10 for Qwen Image 2.1, ~30 for MOSS Image. Ignored when a sigmas input is connected."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "1.0 for turbo/distilled and guidance-embedded models. Above 1.0 every region is evaluated twice per step."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Ignored when a sigmas input is connected."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ignored when a sigmas input is connected - the custom schedule is used as given."}),
                "refine_steps": ("INT", {"default": 8, "min": 0, "max": 10000, "tooltip": "Steps of the detail pass (0 = off). With a sigmas input the detail pass keeps the schedule's step count instead."}),
                "refine_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the picture the detail pass may redo. 0.25-0.4 sharpens and cleans region edges without changing the composition; use 0.45-0.6 together with refine_upscale."}),
                "refine_upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Upscale the latent before the detail pass (1.0 = off). 1.25-1.5 with refine_denoise 0.45-0.6 gives the crispest result."}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"tooltip": "Optional custom sigma schedule (e.g. from a scheduler/sigmas node). Used for the main pass as given; the detail pass reuses it scaled by refine_denoise. Overrides steps, scheduler and denoise."}),
                "preview": ("STAR_PREVIEW", {"tooltip": "Optional ⭐ Star Preview options - shows a live sampling preview on the connected ⭐ Star Preview node (comfyui_starnodes pack)."}),
                "clip": ("CLIP", {"tooltip": "Required for per-box face detailing. Connect the same CLIP as the canvas."}),
                "vae": ("VAE", {"tooltip": "Required for per-box face detailing. Connect the VAE used to decode the result."}),
                "detector_model": (_detector_choices(), {"default": DEFAULT_DETECTOR, "tooltip": "WeVisOne-VL encoder used for per-face alpha mattes on boxes marked 'Face detailer' in the canvas. 2B int8 (2.7 GB, default) is fastest; 4B int8 (4.9 GB) better; 4B bf16 (9.0 GB) best quality. A missing encoder is downloaded automatically from Comfy-Org/WeVisOne-VL on first use."}),
                "face_steps": ("INT", {"default": 12, "min": 0, "max": 1000, "tooltip": "Face refinement steps for checked canvas boxes (0 = off). Uses the main sampler's cfg/sampler/scheduler."}),
                "face_denoise": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Denoise strength of the per-box face refinement."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"
    CATEGORY = "⭐StarNodes/Sampler"

    def sample(self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler,
               denoise, refine_steps, refine_denoise, refine_upscale,
               sigmas=None, preview=None, clip=None, vae=None,
               detector_model=DEFAULT_DETECTOR, face_steps=12, face_denoise=0.7):
        # Region prompts travel in the conditioning, region LoRAs only in the
        # canvas node's model output. Warn when only the prompts arrive - the
        # LoRAs would be silently ignored otherwise.
        for conds in (positive, negative):
            if any(_REGION_KEY in (getattr(hook, "transformers_dict", None) or {})
                   for entry in conds
                   for hook in getattr((entry[1] or {}).get("hooks") if len(entry) > 1 else None, "hooks", ())):
                if not any(key.startswith(f"{_REGION_KEY}_") for key in getattr(model, "injections", {})):
                    logging.warning(
                        "[Star Regional Sampler] the conditioning carries region LoRAs but the model has none - "
                        "connect the canvas node's 'model' output (not the checkpoint loader's) "
                        "to this sampler or all region LoRAs are ignored")
                break

        if preview is not None:
            if apply_star_preview is None:
                logging.warning("[Star Regional Sampler] preview connected but the comfyui_starnodes pack is not available - sampling without preview")
            else:
                model = apply_star_preview(model, preview)

        if sigmas is not None:
            if sigmas.numel() < 2:
                logging.warning("[Star Regional Sampler] sigmas input needs at least 2 values - ignored")
            else:
                logging.info(f"[Star Regional Sampler] custom sigmas: {sigmas.numel() - 1} steps")
                out = _sample_with_sigmas(model, positive, negative, cfg, seed, sampler_name, sigmas, latent)
        if sigmas is None or sigmas.numel() < 2:
            out = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)[0]

        if refine_steps > 0 and refine_denoise > 0.0:
            samples = out["samples"]
            if refine_upscale > 1.0 and samples.ndim == 4:
                # even latent sizes keep 2x2-patch models happy
                width = round(samples.shape[3] * refine_upscale / 2) * 2
                height = round(samples.shape[2] * refine_upscale / 2) * 2
                out = out.copy()
                out["samples"] = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")

            if sigmas is not None and sigmas.numel() >= 2:
                # same schedule, compressed into the low-sigma range
                refine_sigmas = (sigmas.float() * refine_denoise).clone()
                refine_sigmas[-1] = 0.0
                out = _sample_with_sigmas(model, positive, negative, cfg, seed + 1, sampler_name, refine_sigmas, out)
            else:
                out = common_ksampler(model, seed + 1, refine_steps, cfg, sampler_name, scheduler, positive, negative, out, denoise=refine_denoise)[0]

        image_out = None
        face_regions = latent.get("star_face_regions", [])
        if face_regions and face_steps > 0:
            if clip is None or vae is None:
                logging.warning("[Star Regional Sampler] face-detail boxes are selected but CLIP or VAE is not connected - face detailing skipped")
            else:
                image_out, count = detail_regions(
                    vae.decode(out["samples"]), model, clip, vae, face_regions,
                    latent.get("star_face_negative", ""), seed + 10000, face_steps,
                    cfg, sampler_name, scheduler, face_denoise, detector_model)
                if count:
                    out = out.copy()
                    out["samples"] = vae.encode(image_out)

        if image_out is None and vae is not None:
            image_out = vae.decode(out["samples"])

        _check_region_lora_ran(model)
        return (out, image_out)


NODE_CLASS_MAPPINGS = {
    "StarRegionalSampler": StarRegionalSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRegionalSampler": "⭐ Star Regional Sampler",
}
