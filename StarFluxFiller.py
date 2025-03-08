import torch
import logging
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from nodes import common_ksampler, InpaintModelConditioning
from comfy_extras.nodes_flux import FluxGuidance
import comfy.utils
import comfy.samplers
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing

# Utility function for rescaling images
def rescale(samples, width, height, algorithm: str):
    if algorithm == "bislerp":  # convert for compatibility with old workflows
        algorithm = "bicubic"
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    return samples

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

class StarFluxFillerCropAndStitch:
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
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "Process multiple samples in parallel for better GPU utilization"}),
                    "differential_attention": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Use Differential Attention for better results"}),
                    "use_teacache": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Use TeaCache to speed up generation"}),
                }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    
    # Cache for negative prompt encoding to avoid redundant computation
    _neg_cond_cache = {}

    # CropAndStitch helper methods
    def grow_and_blur_mask(self, mask, blur_pixels):
        if blur_pixels > 0.001:
            sigma = blur_pixels / 4
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in growmask:
                mask_np = m.numpy()
                kernel_size = math.ceil(sigma * 1.5 + 1)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dilated_mask = grey_dilation(mask_np, footprint=kernel)
                output = dilated_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

            mask_np = mask.numpy()
            filtered_mask = gaussian_filter(mask_np, sigma=sigma)
            mask = torch.from_numpy(filtered_mask)
            mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    def crop_image_and_mask(self, image, mask):
        # Default CropAndStitch parameters
        context_expand_pixels = 20
        context_expand_factor = 1.0
        fill_mask_holes = True
        blur_mask_pixels = 16.0
        invert_mask = False
        blend_pixels = 16.0
        rescale_algorithm = "bicubic"
        padding = 32
        min_width = 512
        min_height = 512
        max_width = 768
        max_height = 768
        
        # Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                raise ValueError("mask size must match image size")

        # Fill holes if requested
        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Grow and blur mask if requested
        if blur_mask_pixels > 0.001:
            mask = self.grow_and_blur_mask(mask, blur_mask_pixels)

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Use mask as context mask
        context_mask = mask

        # Ensure mask dimensions match image dimensions except channels
        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        
        if initial_height != mask_height or initial_width != mask_width:
            raise ValueError("Image and mask dimensions must match")
        if initial_height != context_mask_height or initial_width != context_mask_width:
            raise ValueError("Image and context mask dimensions must match")

        # Extend image and masks to turn it into a big square in case the context area would go off bounds
        extend_y = (initial_width + 1) // 2 # Intended, extend height by width (turn into square)
        extend_x = (initial_height + 1) // 2 # Intended, extend width by height (turn into square)
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x

        start_y = extend_y
        start_x = extend_x

        available_top = min(start_y, initial_height)
        available_bottom = min(new_height - (start_y + initial_height), initial_height)
        available_left = min(start_x, initial_width)
        available_right = min(new_width - (start_x + initial_width), initial_width)

        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        
        # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
        # Top
        new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(image[:, :available_top, :, :], [1])
        # Bottom
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
        # Left
        new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
        # Right
        new_image[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [2])
        # Top-left corner
        new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
        # Top-right corner
        new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        # Bottom-left corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left, :], [1, 2])
        # Bottom-right corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])

        new_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype) # assume ones in extended image
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask

        blend_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype) # assume zeros in extended image
        blend_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        
        # Mirror blend mask so there's no bleeding of border when blending
        # Top
        blend_mask[:, start_y - available_top:start_y, start_x:start_x + initial_width] = torch.flip(mask[:, :available_top, :], [1])
        # Bottom
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width] = torch.flip(mask[:, -available_bottom:, :], [1])
        # Left
        blend_mask[:, start_y:start_y + initial_height, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x:start_x + available_left], [2])
        # Right
        blend_mask[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [2])
        # Top-left corner
        blend_mask[:, start_y - available_top:start_y, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x:start_x + available_left], [1, 2])
        # Top-right corner
        blend_mask[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width], [1, 2])
        # Bottom-left corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left], [1, 2])
        # Bottom-right corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [1, 2])

        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask

        image = new_image
        mask = new_mask
        context_mask = new_context_mask

        original_image = image
        original_width = image.shape[2]
        original_height = image.shape[1]

        # If there are no non-zero indices in the context_mask, adjust context mask to the whole image
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            context_mask = torch.ones_like(image[:, :, :, 0])
            context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
            context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] += 1.0
            non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_min = max(y_min - int(y_size * (context_expand_factor - 1.0) / 2.0) - context_expand_pixels, 0)
        y_max = min(y_max + int(y_size * (context_expand_factor - 1.0) / 2.0) + context_expand_pixels, height - 1)
        x_min = max(x_min - int(x_size * (context_expand_factor - 1.0) / 2.0) - context_expand_pixels, 0)
        x_max = min(x_max + int(x_size * (context_expand_factor - 1.0) / 2.0) + context_expand_pixels, width - 1)

        # Recalculate x_size and y_size after adjustments
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Pad area to avoid the sampler returning smaller results
        x_min, x_max = self.apply_padding(x_min, x_max, width, padding)
        y_min, y_max = self.apply_padding(y_min, y_max, height, padding)

        # Ensure that context area doesn't go outside of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask_blend = blend_mask[:, y_min:y_max+1, x_min:x_max+1]

        # Grow and blur mask for blend if requested
        if blend_pixels > 0.001:
            cropped_mask_blend = self.grow_and_blur_mask(cropped_mask_blend, blend_pixels)

        # Return crop information and cropped images
        crop_info = {
            'x': x_min, 
            'y': y_min, 
            'original_image': original_image, 
            'cropped_mask_blend': cropped_mask_blend, 
            'rescale_x': 1.0, 
            'rescale_y': 1.0, 
            'start_x': start_x, 
            'start_y': start_y, 
            'initial_width': initial_width, 
            'initial_height': initial_height
        }
        
        return crop_info, cropped_image, cropped_mask

    def stitch_image(self, crop_info, inpainted_image):
        original_image = crop_info['original_image']
        cropped_mask_blend = crop_info['cropped_mask_blend']
        x = crop_info['x']
        y = crop_info['y']
        start_x = crop_info['start_x']
        start_y = crop_info['start_y']
        initial_width = crop_info['initial_width']
        initial_height = crop_info['initial_height']
        
        # Prepare for stitching
        stitched_image = original_image.clone().movedim(-1, 1)
        
        # Composite the inpainted image onto the original
        inpainted_movedim = inpainted_image.movedim(-1, 1)
        
        # Create mask for compositing
        mask = cropped_mask_blend
        mask = mask.to(stitched_image.device)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                                              size=(inpainted_movedim.shape[2], inpainted_movedim.shape[3]), 
                                              mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, inpainted_movedim.shape[0])
        
        # Calculate the bounds for compositing
        left, top = x, y
        right, bottom = left + inpainted_movedim.shape[3], top + inpainted_movedim.shape[2]
        
        # Handle visible portion calculation
        visible_width = stitched_image.shape[3] - left
        visible_height = stitched_image.shape[2] - top
        
        # Adjust mask and source for the visible portion
        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask
        
        # Composite the images
        source_portion = mask * inpainted_movedim[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask * stitched_image[:, :, top:bottom, left:right]
        stitched_image[:, :, top:bottom, left:right] = source_portion + destination_portion
        
        # Crop out from the extended dimensions back to original
        output = stitched_image.movedim(1, -1)
        cropped_output = output[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :]
        
        return cropped_output

    def execute(self, model, clip, text, vae, image, mask, seed, steps, cfg, sampler_name, scheduler, denoise, noise_mask=True, batch_size=1, differential_attention=True, use_teacache=True):
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
                # Import TeaCache functionality
                from custom_nodes.teacache.nodes import TeaCacheForImgGen, teacache_flux_forward
                
                # Create a clone of the model
                teacache_model = model.clone()
                
                # Apply TeaCache with fixed settings (Model Flux, threshold 0.40)
                teacache = TeaCacheForImgGen()
                model = teacache.apply_teacache(teacache_model, "flux", 0.40)[0]
                
                logging.info("TeaCache applied to the model with threshold 0.40")
            except Exception as e:
                logging.warning(f"Failed to apply TeaCache: {str(e)}")
        
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
            
            # Apply CropAndStitch - crop the image and mask
            crop_info, cropped_image, cropped_mask = self.crop_image_and_mask(image, mask)
            
            # Process inpainting with the cropped image and mask
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
                        cropped_image, 
                        vae, 
                        cropped_mask, 
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
                    cropped_image, 
                    vae, 
                    cropped_mask, 
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
            
            # Decode the latent to an image
            decoded_image = vae.decode(latent_result["samples"])
            
            # Stitch the decoded image back into the original
            stitched_image = self.stitch_image(crop_info, decoded_image)
            
            # Encode the stitched image to a latent
            final_latent = {"samples": vae.encode(stitched_image[:,:,:,:3])}
            
        # Clean up any unused tensors to help with memory management
        torch.cuda.empty_cache()
        
        # Return both the stitched image and the latent
        return (stitched_image, final_latent)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FluxFillSampler": StarFluxFillerCropAndStitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFillSampler": "⭐ Star FluxFill Inpainter"
}
