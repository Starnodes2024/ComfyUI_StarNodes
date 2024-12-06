import os
import random
import time
import logging
import folder_paths
import comfy.samplers
import comfy.sample
import torch
from nodes import common_ksampler, CLIPTextEncode
from comfy.utils import ProgressBar
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
from comfy import utils

def conditioning_set_values(cond, values):
    c = []
    for t in cond:
        d = t[1].copy()
        d.update(values)
        n = [t[0], d]
        c.append(n)
    return c

def parse_string_to_list(value):
    """Parse a string into a list of values, handling both numeric and string inputs."""
    if isinstance(value, (int, float)):
        return [int(value) if isinstance(value, int) or value.is_integer() else float(value)]
    value = value.replace("\n", ",").split(",")
    value = [v.strip() for v in value if v.strip()]
    value = [int(float(v)) if float(v).is_integer() else float(v) for v in value if v.replace(".", "").isdigit()]
    return value if value else [0]


class Fluxstarsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "conditioning": ("CONDITIONING", ),
                    "latent": ("LATENT", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                    "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
                    "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
                    "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                    "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                    "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
                }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "conditioning", "latent")
    FUNCTION = "execute"
    CATEGORY = "sampling"

    def execute(self, model, conditioning, latent, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise):
        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        # Parse all input parameters
        steps = parse_string_to_list(steps)
        denoise = parse_string_to_list("1.0" if denoise == "" else denoise)
        guidance = parse_string_to_list("3.5" if guidance == "" else guidance)

        if not is_schnell:
            max_shift = parse_string_to_list("1.15" if max_shift == "" else max_shift)
            base_shift = parse_string_to_list("0.5" if base_shift == "" else base_shift)
        else:
            max_shift = parse_string_to_list("0")
            base_shift = parse_string_to_list("1.0" if base_shift == "" else base_shift)

        # Process latent dimensions
        width = latent["samples"].shape[3] * 8
        height = latent["samples"].shape[2] * 8

        # Initialize output latent
        out_latent = None
        total_samples = len(max_shift) * len(base_shift) * len(guidance) * len(steps) * len(denoise)
        current_sample = 0

        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        # Main sampling loop
        for ms in max_shift:
            for bs in base_shift:
                if is_schnell:
                    work_model = ModelSamplingAuraFlow().patch_aura(model, bs)[0]
                else:
                    work_model = ModelSamplingFlux().patch(model, ms, bs, width, height)[0]
                
                for g in guidance:
                    # Update conditioning with guidance while preserving original structure
                    cond = conditioning_set_values(conditioning, {"guidance": g})
                    
                    for st in steps:
                        for d in denoise:
                            current_sample += 1
                            log = f"Sampling {current_sample}/{total_samples} with seed {seed}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                            logging.info(log)

                            # Create a copy of the input latent to avoid modifying it
                            current_latent = {"samples": latent["samples"].clone()}

                            # Use common_ksampler for more reliable sampling
                            samples = common_ksampler(work_model, seed, st, g, sampler, scheduler, cond, cond, current_latent, denoise=d)[0]

                            if out_latent is None:
                                out_latent = samples
                            else:
                                # Both latents should already be in the correct format
                                out_latent = LatentBatch().batch(out_latent, samples)[0]

                            if total_samples > 1:
                                pbar.update(1)

        return (model, conditioning, out_latent)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "Fluxstarsampler": Fluxstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fluxstarsampler": "⭐ StarSampler FLUX"
}