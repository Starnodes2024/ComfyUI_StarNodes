import os
import random
import time
import logging
import folder_paths
import comfy.samplers
import comfy.sample
import torch
from nodes import common_ksampler, CLIPTextEncode, KSampler
from comfy.utils import ProgressBar
from comfy_extras.nodes_latent import LatentBatch

def parse_string_to_list(value):
    """Parse a string into a list of values, handling both numeric and string inputs."""
    if isinstance(value, (int, float)):
        return [int(value) if isinstance(value, int) or value.is_integer() else float(value)]
    value = value.replace("\n", ",").split(",")
    value = [v.strip() for v in value if v.strip()]
    value = [int(float(v)) if float(v).is_integer() else float(v) for v in value if v.replace(".", "").isdigit()]
    return value if value else [0]

class SDstarsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent": ("LATENT", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "sampling"

    def execute(self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise):
        # Create a copy of the input latent to avoid modifying it
        current_latent = {"samples": latent["samples"].clone()}
        
        # Use common_ksampler for sampling
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, current_latent, denoise=denoise)[0]
        
        return (model, positive, negative, samples)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "SDstarsampler": SDstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDstarsampler": "⭐ StarSampler SD / SDXL"
}