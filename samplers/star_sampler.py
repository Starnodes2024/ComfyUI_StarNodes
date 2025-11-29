import os
import random
import time
import logging
import torch
import comfy.samplers
import comfy.sample
import comfy.model_base
from comfy.utils import ProgressBar

# Try to import from nodes, but handle if not available
try:
    from nodes import common_ksampler, CLIPTextEncode
except ImportError:
    # Fallback if nodes module is not available
    common_ksampler = None
    CLIPTextEncode = None

# Detail Daemon adapted by https://github.com/muerrilla/sd-webui-detail-daemon
# Detail Daemon adapted by https://github.com/Jonseed/ComfyUI-Detail-Daemon

def conditioning_set_values(cond, values):
    """Set values in conditioning."""
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

class StarSampler:
    """
    Unified StarSampler for ComfyUI that works with both Flux and SD/SDXL/SD3.5 models.
    Combines the functionality of both FluxStarSampler and SDStarSampler into one node.
    """
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to use for sampling"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning (prompt)"}),
                "latent": ("LATENT", {"tooltip": "The latent image to denoise"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for sampling"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of sampling steps"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier Free Guidance scale"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler algorithm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta", "tooltip": "Noise schedule"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength"}),
                "vae": ("VAE", {"tooltip": "VAE model for decoding latents"}),
                "decode_image": ("BOOLEAN", {"default": True, "tooltip": "Decode the latent to an image using the VAE"}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (optional, not used for Flux models)"}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Max shift for Flux models (ignored for SD models)"}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Base shift for Flux/AuraFlow models"}),
                "detail_schedule": ("DETAIL_SCHEDULE", {"tooltip": "Optional detail daemon schedule"}),
                "options": ("SIGMAS", {"tooltip": "Optional sigma schedule (e.g. from ⭐ Star FlowMatch Option). When connected, overrides the internal scheduler sigmas for Flux/Aura models."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "VAE", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "image", "vae", "seed")
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes/Sampler"

    def make_detail_schedule(self, steps, detail_amount, detail_start, detail_end, detail_bias, detail_exponent):
        """Create a detail daemon schedule for enhanced sampling."""
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
        """Get detail daemon adjustment for current sigma."""
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
        
        return result

    def is_flux_model(self, model):
        """Check if the model is a Flux/AuraFlow model."""
        try:
            return model.model.model_type == comfy.model_base.ModelType.FLOW
        except:
            return False

    def execute(self, model, positive, latent, seed, steps, cfg, sampler_name, scheduler, denoise, vae, 
                decode_image=True, negative=None, max_shift=1.15, base_shift=0.5, detail_schedule=None, options=None):
        
        # Detect model type
        is_flux = self.is_flux_model(model)
        
        logging.info(f"StarSampler: Model type={'Flux/AuraFlow' if is_flux else 'SD/SDXL/SD3.5'}, steps={steps}, cfg={cfg}")
        
        # For Flux models, use guidance in conditioning instead of CFG
        if is_flux:
            # Import Flux-specific modules
            try:
                from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
                from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
                from comfy_extras.nodes_latent import LatentBatch
            except ImportError:
                raise ImportError("Required ComfyUI modules not found. Make sure ComfyUI is properly installed.")
            
            # Get dimensions
            width = latent["samples"].shape[3] * 8
            height = latent["samples"].shape[2] * 8
            
            # Apply model sampling patches
            model_sampling = ModelSamplingFlux() if not is_flux else ModelSamplingAuraFlow()
            
            if is_flux:
                work_model = model_sampling.patch_aura(model, base_shift)[0]
            else:
                work_model = model_sampling.patch(model, max_shift, base_shift, width, height)[0]
            
            # Setup guidance (use cfg as guidance for Flux)
            cond = conditioning_set_values(positive, {"guidance": cfg})
            
            # If no detail schedule, use fast path
            if detail_schedule is None:
                # Initialize components
                noise_gen = Noise_RandomNoise(seed)
                basic_scheduler = BasicScheduler()
                basic_guider = BasicGuider()
                sampler_advanced = SamplerCustomAdvanced()
                
                # Setup guider
                guider = basic_guider.get_guider(work_model, cond)[0]
                
                # Get sampler and sigmas (allow external sigmas override)
                sampler_obj = comfy.samplers.sampler_object(sampler_name)
                if options is not None:
                    sigmas = options
                else:
                    sigmas = basic_scheduler.get_sigmas(work_model, scheduler, steps, denoise)[0]
                
                # Perform sampling
                out_latent = sampler_advanced.sample(noise_gen, guider, sampler_obj, sigmas, latent)[1]
            else:
                # Use detail schedule path
                current_latent = {"samples": latent["samples"].clone()}
                
                # Initialize sampler
                k_sampler = comfy.samplers.KSampler(work_model, steps=steps, device=latent["samples"].device, 
                                                    sampler=sampler_name, scheduler=scheduler, denoise=denoise)
                
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
                    adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg)
                    return original_forward(x, adjusted_sigma, **extra_args)
                
                # Temporarily replace forward method
                work_model.model.diffusion_model.forward = wrapped_forward
                
                try:
                    # Use common_ksampler for sampling
                    out_latent = common_ksampler(work_model, seed, steps, cfg, sampler_name, scheduler, 
                                                cond, cond, current_latent, denoise=denoise)[0]
                finally:
                    # Restore original forward method
                    work_model.model.diffusion_model.forward = original_forward
        
        else:
            # SD/SDXL/SD3.5 path - uses negative conditioning
            if negative is None:
                # Create empty negative conditioning if not provided
                logging.warning("StarSampler: No negative conditioning provided for SD model. Using empty negative.")
                negative = [[torch.zeros_like(positive[0][0]), positive[0][1].copy()]]
            
            if detail_schedule is not None:
                # Use detail schedule
                current_latent = {"samples": latent["samples"].clone()}
                
                # Initialize sampler
                k_sampler = comfy.samplers.KSampler(model, steps=steps, device=latent["samples"].device, 
                                                    sampler=sampler_name, scheduler=scheduler, denoise=denoise)
                
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
                
                # Store original sigmas
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
                    
                    # Calculate progress based on log space
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
                    
                    # Scale adjustment
                    final_adjustment = dd_adjustment * 2.0
                    adjustment_scale = max(0.0, min(1.0, progress * 2))
                    final_adjustment = final_adjustment * adjustment_scale
                    
                    # Apply the adjustment
                    if final_adjustment > 0.001:
                        adjusted_sigma = sigma * max(1e-06, 1.0 - final_adjustment * cfg)
                    else:
                        adjusted_sigma = sigma
                    
                    return original_forward(x, adjusted_sigma, **extra_args)
                
                # Temporarily replace forward method
                target_module.forward = wrapped_forward
                
                try:
                    # Use common_ksampler for sampling
                    out_latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                                positive, negative, current_latent, denoise=denoise)[0]
                finally:
                    # Restore original forward method
                    target_module.forward = original_forward
            else:
                # Standard sampling without detail daemon
                out_latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            positive, negative, latent, denoise=denoise)[0]
        
        # Decode the latent to an image if requested
        image = None
        if decode_image:
            # Use the updated ComfyUI VAE decode API
            images = vae.decode(out_latent["samples"])
            # Handle batch dimensions (same as standard VAEDecode)
            if len(images.shape) == 5:  # Combine batches
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            image = images
        
        # Return outputs (negative will be None for Flux models if not provided)
        if negative is None:
            negative = positive  # Return positive as negative if not used
        
        return (model, positive, negative, out_latent, image, vae, seed)


# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "StarSampler": StarSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSampler": "⭐ StarSampler (Unified)"
}
