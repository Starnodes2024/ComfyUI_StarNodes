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

# Detail Deamon adapted by https://github.com/muerrilla/sd-webui-detail-daemon
# Detail Deamon adapted by https://github.com/Jonseed/ComfyUI-Detail-Daemon

def parse_string_to_list(value):
    """Parse a string into a list of values, handling both numeric and string inputs."""
    if isinstance(value, (int, float)):
        return [int(value) if isinstance(value, int) or value.is_integer() else float(value)]
    value = value.replace("\n", ",").split(",")
    value = [v.strip() for v in value if v.strip()]
    value = [int(float(v)) if float(v).is_integer() else float(v) for v in value if v.replace(".", "").isdigit()]
    return value if value else [0]

class SDstarsampler:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
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
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "vae": ("VAE", ),
                    "decode_image": ("BOOLEAN", {"default": True, "tooltip": "Decode the latent to an image using the VAE"}),
                },
                "optional": {
                    "detail_schedule": ("DETAIL_SCHEDULE",),
                    "settings_input": ("SDSTAR_SETTINGS",),
                }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "DETAIL_SCHEDULE", "IMAGE", "VAE", "SDSTAR_SETTINGS", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "detail_schedule", "image", "vae", "settings_output", "seed")
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

        print(f"Detail Schedule Indices - start: {start_idx}, mid: {mid_idx}, end: {end_idx}")

        start_values = torch.linspace(0, 1, mid_idx - start_idx + 1)
        start_values = 0.5 * (1 - torch.cos(start_values * torch.pi))
        start_values = start_values**detail_exponent
        if len(start_values) > 0:
            start_values *= detail_amount
            print(f"Start values range: {start_values[0]:.4f} to {start_values[-1]:.4f}")

        end_values = torch.linspace(1, 0, end_idx - mid_idx + 1)
        end_values = 0.5 * (1 - torch.cos(end_values * torch.pi))
        end_values = end_values**detail_exponent
        if len(end_values) > 0:
            end_values *= detail_amount
            print(f"End values range: {end_values[0]:.4f} to {end_values[-1]:.4f}")

        multipliers[start_idx : mid_idx + 1] = start_values
        multipliers[mid_idx : end_idx + 1] = end_values

        print(f"Final multipliers shape: {multipliers.shape}, non-zero elements: {torch.count_nonzero(multipliers)}")
        print(f"Multipliers range: {multipliers.min():.4f} to {multipliers.max():.4f}")
        return multipliers

    def get_dd_schedule(self, sigma, sigmas, dd_schedule):
        # Find the neighboring sigma values
        dists = torch.abs(sigmas - sigma)
        idxlow = torch.argmin(dists)
        nlow = float(sigmas[idxlow])

        # If we're at the last sigma, return the last schedule value
        if idxlow == len(sigmas) - 1:
            return float(dd_schedule[idxlow])

        # Get the high neighbor
        idxhigh = idxlow + 1
        nhigh = float(sigmas[idxhigh])

        # Ratio of how close we are to the high neighbor
        ratio = float((sigma - nlow) / (nhigh - nlow))
        ratio = max(0.0, min(1.0, ratio))  # Clamp between 0 and 1
        
        # Mix the DD schedule high/low items according to the ratio
        result = float(torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], torch.tensor(ratio)).item())
        
        # Log every 5th adjustment to avoid spam
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 5 == 0:
            print(f"DD Schedule - sigma: {sigma:.4f}, adjustment: {result:.4f}, ratio: {ratio:.4f}")
            
        return result

    def execute(self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise, vae, decode_image=True, detail_schedule=None, settings_input=None):
        # Apply settings if provided
        if settings_input is not None:
            from .starsamplersettings import StarSamplerSettings
            settings_manager = StarSamplerSettings()
            # Create a dictionary with the current settings
            current_settings = {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "control_after_generate": False  # Default value
            }
            
            # Apply the input settings
            updated_settings = settings_manager.apply_settings_to_sdstar(current_settings, settings_input)
            
            # Update the local variables
            seed = updated_settings["seed"]
            steps = updated_settings["steps"]
            cfg = updated_settings["cfg"]
            sampler_name = updated_settings["sampler_name"]
            scheduler = updated_settings["scheduler"]
            denoise = updated_settings["denoise"]
        
        print("\n=== Starting SDstarsampler execution ===")
        print(f"Parameters: steps={steps}, cfg={cfg}, sampler={sampler_name}, scheduler={scheduler}, denoise={denoise}")
        if detail_schedule:
            print(f" Detail Daemon Active with Settings: amount={detail_schedule['detail_amount']:.2f}, start={detail_schedule['detail_start']:.2f}, end={detail_schedule['detail_end']:.2f}, bias={detail_schedule['detail_bias']:.2f}, exponent={detail_schedule['detail_exponent']:.2f}")
            
            # Create a copy of the input latent to avoid modifying it
            current_latent = {"samples": latent["samples"].clone()}
            
            # Initialize sampler to get sigmas for detail daemon adjustments
            k_sampler = comfy.samplers.KSampler(model, steps=steps, device=latent["samples"].device, sampler=sampler_name, scheduler=scheduler, denoise=denoise)
            
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
                device=latent["samples"].device
            )
            
            # Store original sigmas and create modified ones
            original_sigmas = k_sampler.sigmas.clone()
            sigmas_cpu = original_sigmas.detach().cpu()
            
            # Store original forward method
            if hasattr(model.model, 'diffusion_model'):
                original_forward = model.model.diffusion_model.forward
                target_module = model.model.diffusion_model
            else:
                original_forward = model.model.forward
                target_module = model.model
            
            def wrapped_forward(x, sigma, **extra_args):
                # Get the maximum sigma value for this batch
                sigma_float = float(sigma.max().detach().cpu())
                
                # Calculate progress based on log space since sigmas are logarithmically distributed
                log_sigma = torch.log(torch.tensor(sigma_float + 1e-10))
                log_sigma_max = torch.log(torch.tensor(1000.0))
                log_sigma_min = torch.log(torch.tensor(0.1))
                
                progress = 1.0 - (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
                progress = float(progress.clamp(0.0, 1.0))
                
                # Get the schedule index based on progress
                schedule_idx = int(progress * (len(detail_schedule_tensor) - 1))
                schedule_idx = max(0, min(schedule_idx, len(detail_schedule_tensor) - 1))
                
                # Get base adjustment from schedule
                dd_adjustment = float(detail_schedule_tensor[schedule_idx])
                
                # Scale adjustment based on progress and make it stronger
                final_adjustment = dd_adjustment * 2.0
                
                # Apply the adjustment
                adjustment_scale = max(0.0, min(1.0, progress * 2))
                final_adjustment = final_adjustment * adjustment_scale
                
                # Ensure the adjustment is significant enough
                if final_adjustment > 0.001:
                    adjusted_sigma = sigma * max(1e-06, 1.0 - final_adjustment * cfg)
                else:
                    adjusted_sigma = sigma
                
                return original_forward(x, adjusted_sigma, **extra_args)
            
            # Temporarily replace forward method
            target_module.forward = wrapped_forward
            
            try:
                # Use common_ksampler for sampling
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, current_latent, denoise=denoise)[0]
            finally:
                # Restore original forward method
                target_module.forward = original_forward
                
            print(" Detail Daemon Sampling Complete")
        else:
            # Standard sampling without detail daemon
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)[0]
            
        # Decode the latent to an image if requested
        image = None
        if decode_image:
            print("Decoding latent to image using VAE")
            image = vae.decode(samples["samples"])
        
        # Create settings output
        settings_output = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "control_after_generate": False  # Default value
        }
            
        return (model, positive, negative, samples, detail_schedule, image, vae, settings_output, seed)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "SDstarsampler": SDstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDstarsampler": "⭐ StarSampler SD (DEPRECATED! please use Unfied Sampler)"
}