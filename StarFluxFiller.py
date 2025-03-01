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
                }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", "image",)
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes"

    def execute(self, model, clip, text, vae, image, mask, seed, steps, cfg, sampler_name, scheduler, denoise, noise_mask=True, decode_image=False):
        # Use default prompt if text input is empty
        if not text.strip():
            text = "A Fluffy Confused Purple Monster with a \"?\" Sign"
        
        # Generate Positive Conditioning from text - more efficient with single tokenize call
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning_pos = [[cond, output]]
        
        # Generate Negative Conditioning with empty string
        tokens = clip.tokenize("")  # Empty string for negative prompt
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning_neg = [[cond, output]]
        
        # Apply FluxGuidance with fixed value of 30 - directly modify conditioning
        conditioning_pos = FluxGuidance().append(conditioning_pos, 30.0)[0]
        
        # Process inpainting - this is the most compute-intensive part
        # Use InpaintModelConditioning to process the image and mask
        conditioning_pos, conditioning_neg, latent = InpaintModelConditioning().encode(
            conditioning_pos, 
            conditioning_neg, 
            image, 
            vae, 
            mask, 
            noise_mask
        )
        
        # Direct reference to latent for sampling - avoid unnecessary cloning when possible
        # Only clone if we need to modify it
        if noise_mask and "noise_mask" in latent:
            # Use direct reference to avoid unnecessary memory operations
            latent_result = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                     conditioning_pos, conditioning_neg, latent, denoise=denoise)[0]
        else:
            # Create a minimal copy if needed
            current_latent = {"samples": latent["samples"]}
            latent_result = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                     conditioning_pos, conditioning_neg, current_latent, denoise=denoise)[0]
        
        # Decode the latent to an image if requested
        if decode_image:
            # Use VAE to decode the latent into an image
            decoded_image = vae.decode(latent_result["samples"])
            if len(decoded_image.shape) == 5:  # Combine batches if needed
                decoded_image = decoded_image.reshape(-1, decoded_image.shape[-3], decoded_image.shape[-2], decoded_image.shape[-1])
        else:
            # Return the original input image when not decoding
            # This ensures we always return a valid IMAGE type
            decoded_image = image
        
        # Return both the latent and the decoded image
        return (latent_result, decoded_image)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FluxFillSampler": StarFluxFiller
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFillSampler": "⭐ Star FluxFill Inpainter"
}
