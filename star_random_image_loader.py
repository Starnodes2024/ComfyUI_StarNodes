import os
import torch
import numpy as np
from PIL import Image, ImageOps
import random

class StarRandomImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "image_path", "seed")
    FUNCTION = "load_random_image"
    CATEGORY = "⭐StarNodes/Image And Latent"
    
    # Add UI display for the seed
    @classmethod
    def EXTRA_OUTPUTS(s):
        return {"seed": {"display": "seed"}}

    @classmethod
    def IS_CHANGED(s, folder, include_subfolders=False, seed=0):
        # If seed is 0, use random behavior (default)
        # Otherwise use the seed for consistent behavior
        if seed == 0:
            return random.random()
        else:
            return seed

    def load_random_image(self, folder, include_subfolders=False, seed=0):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        image_paths = []
        
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder, file))

        if not image_paths:
            raise FileNotFoundError(f"No valid images found in directory '{folder}'.")

        # Sort paths to ensure consistent ordering regardless of filesystem
        image_paths.sort()
        
        # Generate or use provided seed
        used_seed = seed
        if seed == 0:
            used_seed = random.randint(0, 0xffffffffffffffff)
        
        # Use the seed to select an image deterministically
        rng = random.Random(used_seed)
        random_image_path = rng.choice(image_paths)
        
        img = Image.open(random_image_path)
        img = ImageOps.exif_transpose(img)
        
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        height, width = image.shape[1:3]
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")

        return (image, mask, random_image_path, used_seed)

NODE_CLASS_MAPPINGS = {
    "StarRandomImageLoader": StarRandomImageLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRandomImageLoader": "\u2b50 Star Random Image Loader",
}
