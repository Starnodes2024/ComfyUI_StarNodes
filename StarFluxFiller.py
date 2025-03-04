import torch
import logging
import numpy as np
import torch.nn.functional as F
import types
from nodes import common_ksampler, InpaintModelConditioning
from comfy_extras.nodes_flux import FluxGuidance
import node_helpers
import comfy.samplers

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

class DifferentialDiffusion:
    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return model

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)

class StarFluxFiller:
    # Define custom colors for the node
    COLOR = "#19124d"  # Title color
    BGCOLOR = "#3d124d"  # Background color
    CATEGORY = "⭐StarNodes"
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "clip": ("CLIP", ),
                    "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "What you want to inpaint?"}),
                    "vae": ("VAE", ),
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "noise_mask": ("BOOLEAN", {"default": True, "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."}),
                    "decode_image": ("BOOLEAN", {"default": True, "tooltip": "Decode the latent to an image using the VAE"}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "Process multiple samples in parallel for better GPU utilization"}),
                    "differential_attention": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Use Differential Attention for better results"}),
                    "use_teacache": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Use TeaCache to speed up generation"}),
                    "clip_attention_multiply": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Apply attention multipliers to the CLIP model for better results"}),
                }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", "image",)
    FUNCTION = "execute"
    
    # Cache for negative prompt encoding to avoid redundant computation
    _neg_cond_cache = {}
    
    def patch_clip_attention(self, clip):
        """Apply attention multipliers to CLIP model"""
        # Fixed values as specified
        q = 1.20
        k = 1.10
        v = 0.8
        out = 1.25
        
        m = clip.clone()
        sd = m.patcher.model_state_dict()

        for key in sd:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, q)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, k)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, v)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, out)
        return m

    def execute(self, model, clip, text, vae, image, mask, seed, steps, cfg, sampler_name, scheduler, denoise, noise_mask=True, decode_image=False, batch_size=1, differential_attention=True, use_teacache=True, clip_attention_multiply=True):
        # Apply Differential Diffusion if enabled
        if differential_attention:
            try:
                # Apply Differential Diffusion
                diff_diffusion = DifferentialDiffusion()
                model = diff_diffusion.apply(model)
                logging.info("Using differential attention!")
            except Exception as e:
                logging.warning(f"Failed to apply Differential Diffusion: {str(e)}")
        
        # Apply TeaCache if enabled
        if use_teacache:
            try:
                # Apply TeaCache with fixed settings (threshold 0.40)
                model = apply_teacache_patch(model, 0.40)
                logging.info("TeaCache applied to the model with threshold 0.40")
            except Exception as e:
                logging.warning(f"Failed to apply TeaCache: {str(e)}")
        
        # Use default prompt if text input is empty
        if not text.strip():
            text = "A Fluffy Confused Purple Monster with a \"?\" Sign"
        
        # Use torch.no_grad for all inference operations to reduce memory usage and improve speed
        with torch.no_grad():
            # Apply CLIP attention multiply if enabled
            if clip_attention_multiply:
                clip_for_cond = self.patch_clip_attention(clip)
                logging.info("CLIP attention multipliers applied")
            else:
                clip_for_cond = clip
            
            # Generate Positive Conditioning from text - more efficient with single tokenize call
            tokens = clip_for_cond.tokenize(text)
            output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning_pos = [[cond, output]]
            
            # Get negative conditioning from cache if possible
            cache_key = f"{clip_for_cond.__class__.__name__}_{id(clip_for_cond)}"
            if cache_key in self._neg_cond_cache:
                conditioning_neg = self._neg_cond_cache[cache_key]
            else:
                # Generate Negative Conditioning with empty string
                tokens = clip_for_cond.tokenize("")  # Empty string for negative prompt
                output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = output.pop("cond")
                conditioning_neg = [[cond, output]]
                # Store in cache for future use
                self._neg_cond_cache[cache_key] = conditioning_neg
            
            # Apply FluxGuidance with fixed value of 30 - directly modify conditioning
            # Use a constant value to avoid redundant computations
            conditioning_pos = FluxGuidance().append(conditioning_pos, 30.0)[0]
            
            # Process inpainting - optimize for batch processing if applicable
            if batch_size > 1:
                # Expand inputs for batch processing
                batch_latents = []
                
                # Process inpainting for each batch item
                for i in range(batch_size):
                    # Use a different seed for each batch item
                    batch_seed = seed + i
                    
                    # Process this batch item
                    batch_cond_pos, batch_cond_neg, batch_latent = InpaintModelConditioning().encode(
                        conditioning_pos, 
                        conditioning_neg, 
                        image, 
                        vae, 
                        mask, 
                        noise_mask
                    )
                    
                    # Perform sampling for this batch item
                    if noise_mask and "noise_mask" in batch_latent:
                        batch_result = common_ksampler(model, batch_seed, steps, cfg, sampler_name, scheduler, 
                                                batch_cond_pos, batch_cond_neg, batch_latent, denoise=denoise)[0]
                    else:
                        # Avoid unnecessary dictionary creation
                        current_latent = {"samples": batch_latent["samples"]}
                        batch_result = common_ksampler(model, batch_seed, steps, cfg, sampler_name, scheduler, 
                                                batch_cond_pos, batch_cond_neg, current_latent, denoise=denoise)[0]
                    
                    batch_latents.append(batch_result["samples"])
                
                # Combine batch results
                combined_samples = torch.cat(batch_latents, dim=0)
                latent_result = {"samples": combined_samples}
            else:
                # Single sample processing - original flow with optimizations
                # Process inpainting with optimized memory handling
                conditioning_pos, conditioning_neg, latent = InpaintModelConditioning().encode(
                    conditioning_pos, 
                    conditioning_neg, 
                    image, 
                    vae, 
                    mask, 
                    noise_mask
                )
                
                # Optimize latent handling to reduce memory operations
                if noise_mask and "noise_mask" in latent:
                    # Use direct reference to avoid unnecessary memory operations
                    latent_result = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            conditioning_pos, conditioning_neg, latent, denoise=denoise)[0]
                else:
                    # Create a minimal reference instead of a full copy
                    # This avoids unnecessary tensor copying
                    current_latent = {"samples": latent["samples"]}
                    latent_result = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            conditioning_pos, conditioning_neg, current_latent, denoise=denoise)[0]
            
            # Decode the latent to an image if requested - only when needed
            if decode_image:
                # Use VAE to decode the latent into an image with optimized batch handling
                # Process in smaller chunks if the batch is large to avoid OOM errors
                max_batch_size = 4  # Maximum batch size for decoding to avoid OOM
                if latent_result["samples"].shape[0] > max_batch_size:
                    # Process in chunks
                    decoded_chunks = []
                    for i in range(0, latent_result["samples"].shape[0], max_batch_size):
                        chunk = {"samples": latent_result["samples"][i:i+max_batch_size]}
                        decoded_chunk = vae.decode(chunk["samples"])
                        decoded_chunks.append(decoded_chunk)
                    decoded_image = torch.cat(decoded_chunks, dim=0)
                else:
                    # Decode the entire batch at once
                    decoded_image = vae.decode(latent_result["samples"])
            else:
                # Return empty tensor if decoding is not requested
                decoded_image = torch.zeros((1, 3, 64, 64))  # Placeholder empty image
        
        # Clean up any unused tensors to help with memory management
        torch.cuda.empty_cache()
        
        # Return both the latent and the decoded image
        return (latent_result, decoded_image)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FluxFillSampler": StarFluxFiller
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFillSampler": "⭐ Star FluxFill Inpainter"
}
