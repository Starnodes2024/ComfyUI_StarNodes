import os
import random
import time
import logging
import folder_paths
import comfy.samplers
import comfy.sample
import torch
import numpy as np
import torch.nn.functional as F
import types
from nodes import common_ksampler, CLIPTextEncode
from comfy.utils import ProgressBar
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_latent import LatentBatch
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
from comfy import utils

# Detail Deamon adapted by https://github.com/muerrilla/sd-webui-detail-daemon
# Detail Deamon adapted by https://github.com/Jonseed/ComfyUI-Detail-Daemon

# TeaCache implementation
class PatchKeys:
    options_key = "patches_point"
    running_net_model = "running_net_model"
    dit_enter = "patch_dit_enter"
    dit_blocks_before = "patch_dit_blocks_before"
    dit_double_blocks_replace = "patch_dit_double_blocks_replace"
    dit_blocks_transition_replace = "patch_dit_blocks_transition_replace"
    dit_single_blocks_replace = "patch_dit_single_blocks_replace"
    dit_blocks_after_transition_replace = "patch_dit_final_layer_before_replace"
    dit_final_layer_before = "patch_dit_final_layer_before"
    dit_exit = "patch_dit_exit"

def set_model_patch(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = to[options_key].get(name, []) + [patch]

def set_model_patch_replace(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = patch

def add_model_patch_option(model, patch_key):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if patch_key not in to:
        to[patch_key] = {}
    return to[patch_key]

def is_flux_model(model):
    if isinstance(model, comfy.ldm.flux.model.Flux):
        return True
    return False

# TeaCache implementation
tea_cache_key_attrs = "tea_cache_attr"
coefficients_obj = {
    'Flux': [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
}

def get_teacache_global_cache(transformer_options, timesteps):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, "flux_tea_cache"):
        tea_cache = getattr(diffusion_model, "flux_tea_cache", {})
        transformer_options[tea_cache_key_attrs] = tea_cache
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    attrs['step_i'] = timesteps[0].detach().cpu().item()

def tea_cache_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    get_teacache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def tea_cache_patch_blocks_before(img, txt, vec, ids, pe, transformer_options):
    real_model = transformer_options[PatchKeys.running_net_model]
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    step_i = attrs['step_i']
    timestep_start = attrs['timestep_start']
    timestep_end = attrs['timestep_end']
    in_step = timestep_end <= step_i <= timestep_start

    if attrs['rel_l1_thresh'] > 0 and in_step:
        inp = img.clone()
        vec_ = vec.clone()
        coefficient_type = 'Flux'
        
        # Flux model
        double_block_0 = real_model.double_blocks[0]
        img_mod1, img_mod2 = double_block_0.img_mod(vec_)
        modulated_inp = double_block_0.img_norm1(inp)
        modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift
        
        if attrs['cnt'] == 0 or attrs['cnt'] == attrs['total_steps'] - 1:
            should_calc = True
            attrs['accumulated_rel_l1_distance'] = 0
        else:
            coefficients = coefficients_obj[coefficient_type]
            rescale_func = np.poly1d(coefficients)
            attrs['accumulated_rel_l1_distance'] += rescale_func(((modulated_inp - attrs['previous_modulated_input']).abs().mean() / attrs['previous_modulated_input'].abs().mean()).cpu().item())

            if attrs['accumulated_rel_l1_distance'] < attrs['rel_l1_thresh']:
                should_calc = False
            else:
                should_calc = True
                attrs['accumulated_rel_l1_distance'] = 0
        attrs['previous_modulated_input'] = modulated_inp
        attrs['cnt'] += 1
        if attrs['cnt'] == attrs['total_steps']:
            attrs['cnt'] = 0
    else:
        should_calc = True

    attrs['should_calc'] = should_calc

    del real_model
    return img, txt, vec, ids, pe

def tea_cache_patch_double_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if not should_calc:
        img += attrs['previous_residual']
    else:
        attrs['ori_img'] = img.clone()
        img, txt = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_transition_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
    return img

def tea_cache_patch_single_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_after_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args)
    return img

def tea_cache_patch_final_transition_after(img, txt, transformer_options):
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        attrs['previous_residual'] = img - attrs['ori_img']
    return img

def tea_cache_patch_dit_exit(img, transformer_options):
    tea_cache = transformer_options.get(tea_cache_key_attrs, {})
    setattr(transformer_options.get(PatchKeys.running_net_model), "flux_tea_cache", tea_cache)
    return img

def tea_cache_prepare_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj

    # Use cfd_guider.model_options, which is copied from modelPatcher.model_options and will be restored after execution without any unexpected contamination
    temp_options = add_model_patch_option(cfg_guider, tea_cache_key_attrs)
    temp_options['total_steps'] = len(sigmas) - 1
    temp_options['cnt'] = 0
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed)
    finally:
        diffusion_model = cfg_guider.model_patcher.model.diffusion_model
        if hasattr(diffusion_model, "flux_tea_cache"):
            del diffusion_model.flux_tea_cache

    return out

def apply_teacache_patch(model, rel_l1_thresh=0.4):
    model = model.clone()
    patch_key = "tea_cache_wrapper"
    if rel_l1_thresh == 0 or len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0:
        return model

    diffusion_model = model.get_model_object('diffusion_model')
    if not is_flux_model(diffusion_model):
        logging.warning("TeaCache patch is not applied because the model is not Flux.")
        return model

    tea_cache_attrs = add_model_patch_option(model, tea_cache_key_attrs)

    tea_cache_attrs['rel_l1_thresh'] = rel_l1_thresh
    model_sampling = model.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(0.0)
    sigma_end = model_sampling.percent_to_sigma(1.0)
    tea_cache_attrs['timestep_start'] = model_sampling.timestep(sigma_start)
    tea_cache_attrs['timestep_end'] = model_sampling.timestep(sigma_end)

    set_model_patch(model, PatchKeys.options_key, tea_cache_enter, PatchKeys.dit_enter)
    set_model_patch(model, PatchKeys.options_key, tea_cache_patch_blocks_before, PatchKeys.dit_blocks_before)
    set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_double_blocks_replace, PatchKeys.dit_double_blocks_replace)
    set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
    set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
    set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)
    set_model_patch(model, PatchKeys.options_key, tea_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
    set_model_patch(model, PatchKeys.options_key, tea_cache_patch_dit_exit, PatchKeys.dit_exit)

    # Just add it once when connecting in series
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                               patch_key,
                               tea_cache_prepare_wrapper
                               )
    return model

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
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
                "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
                "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
                "use_teacache": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "vae": ("VAE", ),
                "decode_image": ("BOOLEAN", {"default": True, "tooltip": "Decode the latent to an image using the VAE"}),
            },
            "optional": {
                "detail_schedule": ("DETAIL_SCHEDULE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "DETAIL_SCHEDULE", "IMAGE")
    RETURN_NAMES = ("model", "conditioning", "latent", "detail_schedule", "image")
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes"

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

    def execute(self, model, conditioning, latent, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, use_teacache, vae, decode_image=True, detail_schedule=None):
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
        from comfy_extras.nodes_latent import LatentBatch

        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        # Apply TeaCache if enabled and model is Flux
        if use_teacache and not is_schnell:
            try:
                # Apply TeaCache with fixed settings (threshold 0.40)
                model = apply_teacache_patch(model, 0.40)
                logging.info("TeaCache applied to the model with threshold 0.40")
            except Exception as e:
                logging.warning(f"Failed to apply TeaCache: {str(e)}")

        # Parse input parameters
        steps_list = parse_string_to_list(steps)
        denoise = parse_string_to_list("1.0" if denoise == "" else denoise)
        guidance = parse_string_to_list("3.5" if guidance == "" else guidance)

        if not is_schnell:
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
            model_sampling = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()

            # Process model and get sampling components
            if is_schnell:
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
                    if is_schnell:
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
            print("Decoding latent to image using VAE")
            image = vae.decode(out_latent["samples"])

        return (model, conditioning, out_latent, detail_schedule, image)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "Fluxstarsampler": Fluxstarsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fluxstarsampler": "⭐ StarSampler FLUX"
}