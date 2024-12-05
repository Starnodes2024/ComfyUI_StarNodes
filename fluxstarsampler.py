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
                    "seed": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "-1" }),
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

        noise = seed.replace("\n", ",").split(",")
        noise = [random.randint(0, 999999) if "-1" in n else int(n) for n in noise]
        if not noise:
            noise = [random.randint(0, 999999)]

        sampler = [sampler]  # Convert single selection to list for compatibility
        scheduler = [scheduler]  # Convert single selection to list for compatibility

        if steps == "":
            if is_schnell:
                steps = "4"
            else:
                steps = "20"
        steps = parse_string_to_list(steps)

        denoise = "1.0" if denoise == "" else denoise
        denoise = parse_string_to_list(denoise)

        guidance = "3.5" if guidance == "" else guidance
        guidance = parse_string_to_list(guidance)

        if not is_schnell:
            max_shift = "1.15" if max_shift == "" else max_shift
            base_shift = "0.5" if base_shift == "" else base_shift
        else:
            max_shift = "0"
            base_shift = "1.0" if base_shift == "" else base_shift

        max_shift = parse_string_to_list(max_shift)
        base_shift = parse_string_to_list(base_shift)

        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_encoded = conditioning["encoded"]
        else:
            cond_encoded = [conditioning]

        out_latent = None

        basicschedueler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent["samples"].shape[3]*8
        height = latent["samples"].shape[2]*8

        total_samples = len(cond_encoded) * len(noise) * len(max_shift) * len(base_shift) * len(guidance) * len(sampler) * len(scheduler) * len(steps) * len(denoise)
        current_sample = 0
        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        for i in range(len(cond_encoded)):
            conditioning = cond_encoded[i]
            for n in noise:
                randnoise = Noise_RandomNoise(n)
                for ms in max_shift:
                    for bs in base_shift:
                        if is_schnell:
                            work_model = modelsamplingflux.patch_aura(model, bs)[0]
                        else:
                            work_model = modelsamplingflux.patch(model, ms, bs, width, height)[0]
                        for g in guidance:
                            cond = conditioning_set_values(conditioning, {"guidance": g})
                            guider = basicguider.get_guider(work_model, cond)[0]
                            for s in sampler:
                                samplerobj = comfy.samplers.sampler_object(s)
                                for sc in scheduler:
                                    for st in steps:
                                        for d in denoise:
                                            sigmas = basicschedueler.get_sigmas(work_model, sc, st, d)[0]
                                            current_sample += 1
                                            log = f"Sampling {current_sample}/{total_samples} with seed {n}, sampler {s}, scheduler {sc}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                                            logging.info(log)
                                            latent = samplercustomadvanced.sample(randnoise, guider, samplerobj, sigmas, latent)[1]

                                            if out_latent is None:
                                                out_latent = latent
                                            else:
                                                out_latent = latentbatch.batch(out_latent, latent)[0]

                                            if total_samples > 1:
                                                pbar.update(1)
                                            
                                            # Generate new random seed for next iteration if seed is "-1"
                                            if "-1" in seed:
                                                n = random.randint(0, 999999)
                                                randnoise = Noise_RandomNoise(n)

        return (model, conditioning, out_latent)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "Fluxstarsampler": Fluxstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fluxstarsampler": "⭐StarSampler FLUX"
}