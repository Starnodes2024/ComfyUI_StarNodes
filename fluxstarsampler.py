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

# Detail Deamon adapted by https://github.com/muerrilla/sd-webui-detail-daemon
# Detail Deamon adapted by https://github.com/Jonseed/ComfyUI-Detail-Daemon

class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleInputs(dict):
    """A special class to make flexible node inputs."""
    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type, )

    def __contains__(self, key):
        return True

any_type = AnyType("*")

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
        return {
            "required": {
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
            },
            "optional": {
                "detail_schedule": ("DETAIL_SCHEDULE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "conditioning", "latent")
    FUNCTION = "execute"
    CATEGORY = "sampling"

    def make_detail_schedule(self, steps, detail_amount, detail_start, detail_end, detail_bias, detail_exponent):
        start = min(detail_start, detail_end)
        mid = start + detail_bias * (detail_end - start)
        multipliers = torch.zeros(steps + 1)  

        start_idx, mid_idx, end_idx = [
            int(round(x * steps)) for x in [start, mid, detail_end]
        ]

        if start_idx == mid_idx:
            mid_idx = start_idx + 1
        if mid_idx == end_idx:
            end_idx = mid_idx + 1

        # Ensure we don't exceed array bounds
        end_idx = min(end_idx, steps)
        mid_idx = min(mid_idx, end_idx - 1)
        start_idx = min(start_idx, mid_idx)

        start_values = torch.linspace(0, 1, mid_idx - start_idx + 1)
        start_values = 0.5 * (1 - torch.cos(start_values * torch.pi))
        start_values = start_values**detail_exponent
        if len(start_values) > 0:
            start_values *= detail_amount

        end_values = torch.linspace(1, 0, end_idx - mid_idx + 1)
        end_values = 0.5 * (1 - torch.cos(end_values * torch.pi))
        end_values = end_values**detail_exponent
        if len(end_values) > 0:
            end_values *= detail_amount

        multipliers[start_idx : mid_idx + 1] = start_values
        multipliers[mid_idx : end_idx + 1] = end_values

        return multipliers

    def get_dd_schedule(self, sigma, sigmas, dd_schedule):
        # Find the neighboring sigma values
        dists = torch.abs(sigmas - sigma)
        idxlow = torch.argmin(dists)
        nlow = float(sigmas[idxlow])
        
        # If we're at the last sigma, return the last schedule value
        if idxlow == len(sigmas) - 1:
            return dd_schedule[idxlow]
            
        # Get the high neighbor
        idxhigh = idxlow + 1
        nhigh = float(sigmas[idxhigh])
        
        # If we're closer to the low neighbor, just return its value
        if abs(sigma - nlow) < abs(nhigh - nlow) * 1e-3:
            return dd_schedule[idxlow]
            
        # Ratio of how close we are to the high neighbor
        ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
        # Mix the DD schedule high/low items according to the ratio
        return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()

    def create_model_wrapper(self, model, sigmas, detail_schedule, guidance_scale):
        sigmas_cpu = sigmas.detach().clone().cpu()
        sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05
        
        def wrapper(x, sigma, **extra_args):
            sigma_float = float(sigma.max().detach().cpu())
            if not (sigma_min <= sigma_float <= sigma_max):
                return model(x, sigma, **extra_args)
                
            dd_adjustment = self.get_dd_schedule(sigma_float, sigmas_cpu, detail_schedule) * 0.1
            adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * guidance_scale)
            return model(x, adjusted_sigma, **extra_args)
            
        # Copy all model attributes that might be needed
        model_attrs = [
            'inner_model', 'sigmas', 'model', 'model_type', 'get_model_object',
            'latent_channels', 'latent_format', 'model_size', 'model_type',
            'is_adm', 'is_sdxl', 'is_sd2', 'is_sd1', 'is_v2', 'patcher',
            'clip', 'loaded_model'
        ]
        
        for k in model_attrs:
            if hasattr(model, k):
                setattr(wrapper, k, getattr(model, k))
        
        # Add model object methods
        def get_model_object(self=None):
            return model
        
        wrapper.get_model_object = get_model_object
        wrapper.model_type = getattr(model, 'model_type', None)
        wrapper.model = model
        
        # Set default latent channels if not present
        if not hasattr(wrapper, 'latent_channels'):
            wrapper.latent_channels = 4  # Default for SD models
                
        return wrapper

    def execute(self, model, conditioning, latent, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, detail_schedule=None):
        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        # Parse all input parameters
        steps_list = parse_string_to_list(steps)
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
        total_samples = len(max_shift) * len(base_shift) * len(guidance) * len(steps_list) * len(denoise)
        current_sample = 0

        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        # Sampler name mapping
        sampler_map = {
            'euler': 'EulerSampler',
            'euler_ancestral': 'EulerAncestralSampler',
            'heun': 'HeunSampler',
            'dpm_2': 'DPMSolver2Sampler',
            'dpm_2_ancestral': 'DPMSolver2AncestralSampler',
            'lms': 'LMSSampler',
            'dpm_fast': 'DPMSolverFastSampler',
            'dpm_adaptive': 'DPMSolverAdaptiveSampler',
            'dpmpp_2s_ancestral': 'DPMppSDE2SAncestralSampler',
            'dpmpp_sde': 'DPMppSDESampler',
            'dpmpp_2m': 'DPMpp2MSampler',
            'ddim': 'DDIMSampler',
            'uni_pc': 'UniPCSampler',
            'uni_pc_bh2': 'UniPCBh2Sampler'
        }

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
                    
                    for st in steps_list:
                        for d in denoise:
                            current_sample += 1
                            log = f"Sampling {current_sample}/{total_samples} with seed {seed}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                            logging.info(log)

                            # Create a copy of the input latent to avoid modifying it
                            current_latent = {"samples": latent["samples"].clone()}
                            
                            # Initialize sampler to get sigmas for detail daemon adjustments
                            k_sampler = comfy.samplers.KSampler(work_model, steps=st, device=latent["samples"].device, sampler=sampler, scheduler=scheduler, denoise=d)
                            
                            if detail_schedule is not None:
                                # Create detail schedule
                                detail_schedule_tensor = torch.tensor(
                                    self.make_detail_schedule(
                                        len(k_sampler.sigmas) - 1,
                                        detail_schedule["detail_amount"],
                                        detail_schedule["detail_start"],
                                        detail_schedule["detail_end"],
                                        detail_schedule["detail_bias"],
                                        detail_schedule["detail_exponent"]
                                    ),
                                    dtype=torch.float32,
                                    device="cpu"
                                )
                                
                                # Store original sigmas and create modified ones
                                original_sigmas = k_sampler.sigmas.clone()
                                sigmas_cpu = original_sigmas.detach().cpu()
                                sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05
                                
                                # Store original forward method
                                original_forward = work_model.model.diffusion_model.forward
                                
                                def wrapped_forward(x, sigma, **extra_args):
                                    sigma_float = float(sigma.max().detach().cpu())
                                    if not (sigma_min <= sigma_float <= sigma_max):
                                        return original_forward(x, sigma, **extra_args)
                                    dd_adjustment = self.get_dd_schedule(sigma_float, sigmas_cpu, detail_schedule_tensor) * 0.1
                                    adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * g)
                                    return original_forward(x, adjusted_sigma, **extra_args)
                                
                                # Temporarily replace forward method
                                work_model.model.diffusion_model.forward = wrapped_forward
                                
                                try:
                                    # Use common_ksampler for sampling
                                    samples = common_ksampler(work_model, seed, st, g, sampler, scheduler, cond, cond, current_latent, denoise=d)[0]
                                finally:
                                    # Restore original forward method
                                    work_model.model.diffusion_model.forward = original_forward
                            else:
                                # Use common_ksampler without detail daemon
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