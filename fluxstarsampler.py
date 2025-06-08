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
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleInputs(dict):
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
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
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "conditioning": ("CONDITIONING", ),
                "latent": ("LATENT", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
                "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
                "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),

                "vae": ("VAE", ),
                "decode_image": ("BOOLEAN", {"default": True, "tooltip": "Decode the latent to an image using the VAE"}),
            },
            "optional": {
                "detail_schedule": ("DETAIL_SCHEDULE",),
                "settings_input": ("FLUXSTAR_SETTINGS",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "DETAIL_SCHEDULE", "IMAGE", "VAE", "FLUXSTAR_SETTINGS", "INT")
    RETURN_NAMES = ("model", "conditioning", "latent", "detail_schedule", "image", "vae", "settings_output", "seed")
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes/Sampler"

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
        ratio = (sigma - nlow) / (nhigh - nlow)
        ratio = max(0.0, min(1.0, ratio))  # Clamp between 0 and 1
        # Mix the DD schedule high/low items according to the ratio
        return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], torch.tensor(ratio)).item()

    def execute(self, model, conditioning, latent, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, use_teacache, teacache_threshold, teacache_max_skip, vae, decode_image=True, detail_schedule=None, settings_input=None):
        # Apply settings if provided
        if settings_input is not None:
            from .starsamplersettings import StarSamplerSettings
            settings_manager = StarSamplerSettings()
            # Create a dictionary with the current settings
            current_settings = {
                "seed": seed,
                "sampler": sampler,
                "scheduler": scheduler,
                "steps": steps,
                "guidance": guidance,
                "max_shift": max_shift,
                "base_shift": base_shift,
                "denoise": denoise,
                "use_teacache": use_teacache,
                "control_after_generate": False  # Default value
            }
            
            # Apply the input settings
            updated_settings = settings_manager.apply_settings_to_fluxstar(current_settings, settings_input)
            
            # Update the local variables with proper type conversion
            try:
                # Handle seed conversion
                if isinstance(updated_settings["seed"], str):
                    # Check if it's a list-like string
                    if updated_settings["seed"].startswith('[') and updated_settings["seed"].endswith(']'):
                        # Extract the first value from the list-like string
                        seed_str = updated_settings["seed"].strip('[]').split(',')[0].strip()
                        seed = int(seed_str)
                    else:
                        seed = int(updated_settings["seed"])
                else:
                    seed = updated_settings["seed"]
                
                # Handle string values for sampler and scheduler
                sampler = updated_settings["sampler"]
                scheduler = updated_settings["scheduler"]
                
                # Handle steps conversion
                if isinstance(updated_settings["steps"], str):
                    # Check if it's a list-like string
                    if updated_settings["steps"].startswith('[') and updated_settings["steps"].endswith(']'):
                        # Extract the first value from the list-like string
                        steps_str = updated_settings["steps"].strip('[]').split(',')[0].strip()
                        steps = int(steps_str)
                    else:
                        steps = int(updated_settings["steps"])
                else:
                    steps = updated_settings["steps"]
                
                # Handle guidance conversion
                if isinstance(updated_settings["guidance"], str):
                    # Check if it's a list-like string
                    if updated_settings["guidance"].startswith('[') and updated_settings["guidance"].endswith(']'):
                        # Extract the first value from the list-like string
                        guidance_str = updated_settings["guidance"].strip('[]').split(',')[0].strip()
                        guidance = float(guidance_str)
                    else:
                        guidance = float(updated_settings["guidance"])
                else:
                    guidance = updated_settings["guidance"]
                
                # Handle max_shift conversion
                if isinstance(updated_settings["max_shift"], str):
                    # Check if it's a list-like string
                    if updated_settings["max_shift"].startswith('[') and updated_settings["max_shift"].endswith(']'):
                        # Extract the first value from the list-like string
                        max_shift_str = updated_settings["max_shift"].strip('[]').split(',')[0].strip()
                        max_shift = float(max_shift_str)
                    else:
                        max_shift = float(updated_settings["max_shift"])
                else:
                    max_shift = updated_settings["max_shift"]
                
                # Handle base_shift conversion
                if isinstance(updated_settings["base_shift"], str):
                    # Check if it's a list-like string
                    if updated_settings["base_shift"].startswith('[') and updated_settings["base_shift"].endswith(']'):
                        # Extract the first value from the list-like string
                        base_shift_str = updated_settings["base_shift"].strip('[]').split(',')[0].strip()
                        base_shift = float(base_shift_str)
                    else:
                        base_shift = float(updated_settings["base_shift"])
                else:
                    base_shift = updated_settings["base_shift"]
                
                # Handle denoise conversion
                if isinstance(updated_settings["denoise"], str):
                    # Check if it's a list-like string
                    if updated_settings["denoise"].startswith('[') and updated_settings["denoise"].endswith(']'):
                        # Extract the first value from the list-like string
                        denoise_str = updated_settings["denoise"].strip('[]').split(',')[0].strip()
                        denoise = float(denoise_str)
                    else:
                        denoise = float(updated_settings["denoise"])
                else:
                    denoise = updated_settings["denoise"]
                
                # Handle use_teacache conversion
                if isinstance(updated_settings["use_teacache"], str):
                    # Check if it's a list-like string
                    if updated_settings["use_teacache"].startswith('[') and updated_settings["use_teacache"].endswith(']'):
                        # Extract the first value from the list-like string
                        use_teacache_str = updated_settings["use_teacache"].strip('[]').split(',')[0].strip()
                        use_teacache = use_teacache_str.lower() == "true"
                    else:
                        use_teacache = updated_settings["use_teacache"].lower() == "true"
                else:
                    use_teacache = bool(updated_settings["use_teacache"])
            except Exception as e:
                print(f"Error converting settings: {str(e)}")
                # Fall back to original values if conversion fails
                seed = seed
                sampler = sampler
                scheduler = scheduler
                steps = steps
                guidance = guidance
                max_shift = max_shift
                base_shift = base_shift
                denoise = denoise
        
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
        from comfy_extras.nodes_latent import LatentBatch

        # Parse input parameters
        steps_list = parse_string_to_list(steps)
        denoise = parse_string_to_list("1.0" if denoise == "" else denoise)
        guidance = parse_string_to_list("3.5" if guidance == "" else guidance)

        if not model.model.model_type == comfy.model_base.ModelType.FLOW:
            max_shift = parse_string_to_list("1.15" if max_shift == "" else max_shift)
            base_shift = parse_string_to_list("0.5" if base_shift == "" else base_shift)
        else:
            max_shift = parse_string_to_list("0")
            base_shift = parse_string_to_list("1.0" if base_shift == "" else base_shift)

        # Get dimensions
        width = latent["samples"].shape[3] * 8
        height = latent["samples"].shape[2] * 8

        # If no detail schedule is connected, use the fast path
        if detail_schedule is None:
            # Initialize components for fast path
            noise_gen = Noise_RandomNoise(seed)
            basic_scheduler = BasicScheduler()
            basic_guider = BasicGuider()
            sampler_advanced = SamplerCustomAdvanced()
            model_sampling = ModelSamplingFlux() if not model.model.model_type == comfy.model_base.ModelType.FLOW else ModelSamplingAuraFlow()

            # Process model and get sampling components
            if model.model.model_type == comfy.model_base.ModelType.FLOW:
                work_model = model_sampling.patch_aura(model, base_shift[0])[0]
            else:
                work_model = model_sampling.patch(model, max_shift[0], base_shift[0], width, height)[0]

            # Setup guidance
            cond = conditioning_set_values(conditioning, {"guidance": guidance[0]})
            guider = basic_guider.get_guider(work_model, cond)[0]

            # Get sampler and sigmas
            sampler_obj = comfy.samplers.sampler_object(sampler)
            sigmas = basic_scheduler.get_sigmas(work_model, scheduler, steps_list[0], denoise[0])[0]

            # Perform sampling
            out_latent = sampler_advanced.sample(noise_gen, guider, sampler_obj, sigmas, latent)[1]

        else:
            # Use detail schedule path
            out_latent = None
            total_samples = len(max_shift) * len(base_shift) * len(guidance) * len(steps_list) * len(denoise)
            current_sample = 0

            if total_samples > 1:
                pbar = ProgressBar(total_samples)

            # Main sampling loop with detail schedule
            for ms in max_shift:
                for bs in base_shift:
                    if model.model.model_type == comfy.model_base.ModelType.FLOW:
                        work_model = ModelSamplingAuraFlow().patch_aura(model, bs)[0]
                    else:
                        work_model = ModelSamplingFlux().patch(model, ms, bs, width, height)[0]
                    
                    for g in guidance:
                        cond = conditioning_set_values(conditioning, {"guidance": g})
                        
                        for st in steps_list:
                            for d in denoise:
                                current_sample += 1
                                log = f"Sampling {current_sample}/{total_samples} with seed {seed}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                                logging.info(log)

                                # Create a copy of the input latent
                                current_latent = {"samples": latent["samples"].clone()}
                                
                                # Initialize sampler
                                k_sampler = comfy.samplers.KSampler(work_model, steps=st, device=latent["samples"].device, sampler=sampler, scheduler=scheduler, denoise=d)
                                
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
                                
                                # Store original sigmas
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

                                if out_latent is None:
                                    out_latent = samples
                                else:
                                    out_latent = LatentBatch().batch(out_latent, samples)[0]

                                if total_samples > 1:
                                    pbar.update(1)
        
        # Decode the latent to an image if requested
        image = None
        if decode_image:
            image = vae.decode(out_latent["samples"])

        # Create settings output 
        settings_output = {
            "seed": seed,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "guidance": guidance,
            "max_shift": max_shift,
            "base_shift": base_shift,
            "denoise": denoise,
            "use_teacache": use_teacache,
            "teacache_threshold": teacache_threshold,
            "teacache_max_skip": teacache_max_skip,
            "control_after_generate": False  # Default value
        }
        
        # Convert any non-serializable types to strings to ensure proper JSON serialization
        for key, value in settings_output.items():
            if not isinstance(value, (int, float, bool, str, type(None))):
                settings_output[key] = str(value)

        return (model, conditioning, out_latent, detail_schedule, image, vae, settings_output, seed)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "Fluxstarsampler": Fluxstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fluxstarsampler": "⭐ StarSampler FLUX"
}