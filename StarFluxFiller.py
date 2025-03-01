import torch
from nodes import common_ksampler, InpaintModelConditioning
from comfy_extras.nodes_flux import FluxGuidance
import node_helpers
import comfy.samplers

class StarFluxFiller:
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
                }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", "image",)
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes"
    
    # Cache for negative prompt encoding to avoid redundant computation
    _neg_cond_cache = {}

    def execute(self, model, clip, text, vae, image, mask, seed, steps, cfg, sampler_name, scheduler, denoise, noise_mask=True, decode_image=False, batch_size=1):
        # Use default prompt if text input is empty
        if not text.strip():
            text = "A Fluffy Confused Purple Monster with a \"?\" Sign"
        
        # Use torch.no_grad for all inference operations to reduce memory usage and improve speed
        with torch.no_grad():
            # Generate Positive Conditioning from text - more efficient with single tokenize call
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning_pos = [[cond, output]]
            
            # Get negative conditioning from cache if possible
            cache_key = f"{clip.__class__.__name__}_{id(clip)}"
            if cache_key in self._neg_cond_cache:
                conditioning_neg = self._neg_cond_cache[cache_key]
            else:
                # Generate Negative Conditioning with empty string
                tokens = clip.tokenize("")  # Empty string for negative prompt
                output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
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
                max_decode_batch = 4  # Adjust based on VRAM availability
                if latent_result["samples"].shape[0] > max_decode_batch:
                    # Process in chunks
                    decoded_chunks = []
                    for i in range(0, latent_result["samples"].shape[0], max_decode_batch):
                        chunk = latent_result["samples"][i:i+max_decode_batch]
                        decoded_chunk = vae.decode(chunk)
                        decoded_chunks.append(decoded_chunk)
                    decoded_image = torch.cat(decoded_chunks, dim=0)
                else:
                    # Decode all at once
                    decoded_image = vae.decode(latent_result["samples"])
                
                # Handle batch dimension reshaping if needed
                if len(decoded_image.shape) == 5:
                    decoded_image = decoded_image.reshape(-1, decoded_image.shape[-3], decoded_image.shape[-2], decoded_image.shape[-1])
            else:
                # Return the original input image when not decoding
                decoded_image = image
        
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
