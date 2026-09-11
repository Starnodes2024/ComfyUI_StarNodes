import torch
import torch.nn.functional as F
import node_helpers
#bugfix 2.0.1
class StarFlux2Conditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "text": ("STRING", {"multiline": True, "default": "Your prompt here..."}),
                "join_references": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("POS", "NEG", "GRID_IMAGE")
    FUNCTION = "execute"
    CATEGORY = "⭐StarNodes/Conditioning"
    OUTPUT_IS_LIST = (False, False, False)

    def scale_image_to_megapixels(self, image, target_megapixels=1.0):
        """
        Resizes the input image to a target megapixel count while ensuring 
        dimensions are multiples of 16 for VAE stability.
        """
        # Image shape is [B, H, W, C], convert to [B, C, H, W] for torch
        samples = image.movedim(-1, 1)
        _, _, h, w = samples.shape
        
        current_pixels = h * w
        target_pixels = target_megapixels * 1024 * 1024
        
        # Calculate scale factor to hit target pixel count
        scale_factor = (target_pixels / current_pixels) ** 0.5
        
        # Ensure new dimensions are multiples of 16
        new_h = int(round(h * scale_factor / 16) * 16)
        new_w = int(round(w * scale_factor / 16) * 16)
        
        # Perform high-quality bicubic interpolation (Lanczos-like)
        resized = F.interpolate(samples, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        # Return back to [B, H, W, C]
        return resized.movedim(1, -1)

    def create_image_grid(self, images):
        """
        Creates a 2x2 grid layout from multiple images.
        Each cell is 1024x1024 with aspect ratio preserved and white padding.
        Empty cells are filled with white.
        """
        num_images = len(images)
        if num_images == 0:
            return None
        if num_images == 1:
            return images[0]
        
        # Fixed cell size
        cell_size = 1024
        
        # Always create a 2x2 grid
        rows, cols = 2, 2
        
        # Process each image to fit in 1024x1024 with white padding
        processed_images = []
        for img in images:
            # Take only the first image from the batch [B, H, W, C] -> [H, W, C]
            single_img = img[0]
            
            h, w, c = single_img.shape
            
            # Calculate scaling to fit within 1024x1024 while preserving aspect ratio
            scale = min(cell_size / h, cell_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize image
            img_tensor = single_img.unsqueeze(0).movedim(-1, 1)  # [1, C, H, W]
            resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
            resized = resized.movedim(1, -1)[0]  # [H, W, C]
            
            # Create white canvas 1024x1024
            canvas = torch.ones((cell_size, cell_size, c), dtype=single_img.dtype, device=single_img.device)
            
            # Calculate padding to center the image
            pad_top = (cell_size - new_h) // 2
            pad_left = (cell_size - new_w) // 2
            
            # Place resized image on white canvas
            canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
            
            processed_images.append(canvas)
        
        # Fill remaining slots with white images
        while len(processed_images) < rows * cols:
            white_img = torch.ones((cell_size, cell_size, 3), dtype=processed_images[0].dtype, device=processed_images[0].device)
            processed_images.append(white_img)
        
        # Create 2x2 grid
        top_row = torch.cat([processed_images[0], processed_images[1]], dim=1)  # Concatenate horizontally
        bottom_row = torch.cat([processed_images[2], processed_images[3]], dim=1)  # Concatenate horizontally
        grid = torch.cat([top_row, bottom_row], dim=0)  # Concatenate vertically
        
        # Add batch dimension back [H, W, C] -> [1, H, W, C]
        grid = grid.unsqueeze(0)
        
        return grid

    def execute(self, clip, vae, text, join_references, **kwargs):
        # 1. Encode Positive Text Prompt
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        # Construct the standard ComfyUI conditioning structure
        pos_conditioning = [[cond, {"pooled_output": pooled}]]

        # 2. Encode Negative Prompt (Empty string as default for Flux/SDXL workflows)
        tokens_neg = clip.tokenize("")
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        neg_conditioning = [[cond_neg, {"pooled_output": pooled_neg}]]

        # 3. Collect all provided reference images
        images = []
        
        # Always process image_1 separately if provided
        image_1 = kwargs.get("image_1")
        if image_1 is not None:
            images.append((1, image_1))
        
        # Collect images 2-5 for potential grid creation
        additional_images = []
        for i in range(2, 6):
            img_key = f"image_{i}"
            img = kwargs.get(img_key)
            if img is not None:
                additional_images.append(img)
        
        # 4. Process reference images based on join_references setting
        ref_latents = []
        grid_output = None
        
        # Process image_1 if provided
        if image_1 is not None:
            scaled_img = self.scale_image_to_megapixels(image_1, 1.0)
            latent = vae.encode(scaled_img[:,:,:,:3])
            ref_latents.append(latent)
        
        # Process additional images (2-5)
        if len(additional_images) > 0:
            if join_references:
                # Create grid from additional images
                grid_image = self.create_image_grid(additional_images)
                if grid_image is not None:
                    # Store the grid for output
                    grid_output = grid_image
                    # Resize grid to 1MP and encode
                    scaled_grid = self.scale_image_to_megapixels(grid_image, 1.0)
                    latent = vae.encode(scaled_grid[:,:,:,:3])
                    ref_latents.append(latent)
            else:
                # Process each image separately
                for img in additional_images:
                    scaled_img = self.scale_image_to_megapixels(img, 1.0)
                    latent = vae.encode(scaled_img[:,:,:,:3])
                    ref_latents.append(latent)

        # 5. Inject Latents into Conditioning using node_helpers
        if len(ref_latents) > 0:
            pos_conditioning = node_helpers.conditioning_set_values(
                pos_conditioning,
                {"reference_latents": ref_latents},
                append=True
            )

        # 6. Return conditioning and grid image (if created)
        if grid_output is None:
            # Create empty white image if no grid was created
            grid_output = torch.ones((1, 64, 64, 3), dtype=torch.float32)
        
        return (pos_conditioning, neg_conditioning, grid_output)

NODE_CLASS_MAPPINGS = {
    "StarFlux2Conditioner": StarFlux2Conditioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFlux2Conditioner": "⭐ Star Flux2 Conditioner"
}
