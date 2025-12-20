import math
import json
import os
import torch
from PIL import Image

CATEGORY = '⭐StarNodes/Image And Latent'
SDRATIOS_PATH = os.path.join(os.path.dirname(__file__), 'sdratios.json')

class Starnodes_Aspect_Ratio_Advanced:
    """
    Node that takes an image input and outputs its aspect ratio as a string (e.g. '3x4'),
    reduced to the smallest integer ratio. Additionally, outputs latents for Flux/SDXL and SD3.5.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Load aspect ratios from sdratios.json, excluding 'Free Ratio'
        with open(SDRATIOS_PATH, 'r', encoding='utf-8') as f:
            ratios_data = json.load(f)["ratios"]
        ratio_choices = [k for k in ratios_data.keys() if k != "Free Ratio"]

        megapixel_choices = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
        megapixel_choices_str = [str(mp) for mp in megapixel_choices]

        return {
            "optional": {
                "image": ("IMAGE",),
            },
            "required": {
                "aspect_ratio": (ratio_choices, ),
                "megapixel": (megapixel_choices_str, {"default": "1.0"}),
                "latent_channels": (["SDXL / FLUX (4ch)", "SD3.5 (4ch)", "FLUX 2 (128ch)"], {"default": "SDXL / FLUX (4ch)"}),
                "use_nearest_from_image": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "LATENT")
    RETURN_NAMES = ("width", "height", "Resolution", "latent")
    FUNCTION = "get_resolution"
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def get_resolution(self, aspect_ratio, megapixel, latent_channels, use_nearest_from_image=False, image=None, batch_size=1):
        # Load ratios from file
        with open(SDRATIOS_PATH, 'r', encoding='utf-8') as f:
            ratios_data = json.load(f)["ratios"]
        ratios = {k: v for k, v in ratios_data.items() if k != "Free Ratio"}

        def get_ratio_dims(ratio_key):
            r = ratios[ratio_key]
            return int(r["width"]), int(r["height"])

        selected_ratio_key = aspect_ratio
        if use_nearest_from_image and image is not None:
            # Get image width/height
            if hasattr(image, 'shape'):
                if len(image.shape) == 4:
                    _, h, w, _ = image.shape
                elif len(image.shape) == 3:
                    h, w, _ = image.shape
                else:
                    h, w = image.shape
            elif isinstance(image, Image.Image):
                w, h = image.size
            else:
                raise ValueError(f"Unsupported image type for aspect ratio calculation: {type(image)}")
            img_ratio = w / h
            min_diff = float('inf')
            for k, v in ratios.items():
                rw, rh = int(v["width"]), int(v["height"])
                r_ratio = rw / rh
                diff = abs(img_ratio - r_ratio)
                if diff < min_diff:
                    min_diff = diff
                    selected_ratio_key = k
        
        # Get base dimensions from the selected ratio
        base_width, base_height = get_ratio_dims(selected_ratio_key)
        
        # Convert megapixel from string to float
        megapixel_value = float(megapixel)
        
        # If megapixel is 1.0, use the exact dimensions from the JSON file
        if megapixel_value == 1.0:
            width, height = base_width, base_height
        else:
            # Calculate the aspect ratio
            aspect = base_width / base_height
            
            # Calculate new dimensions based on megapixels
            # Formula: width * height = megapixels * 1,000,000
            # and width / height = aspect
            # Therefore: width = sqrt(megapixels * 1,000,000 * aspect)
            #            height = width / aspect
            target_pixels = megapixel_value * 1000000
            width = int(math.sqrt(target_pixels * aspect))
            height = int(width / aspect)
            
            # Ensure width and height are divisible by 8 for latent space
            width = width - (width % 8)
            height = height - (height % 8)
        
        resolution_str = f"{width} x {height}"

        if latent_channels == "FLUX 2 (128ch)":
            width_latent = width - (width % 16)
            height_latent = height - (height % 16)
            latent = torch.zeros([batch_size, 128, height_latent // 16, width_latent // 16])
        else:
            width_latent = width - (width % 8)
            height_latent = height - (height % 8)
            latent = torch.zeros([batch_size, 4, height_latent // 8, width_latent // 8])

        return (width, height, resolution_str, {"samples": latent})

NODE_CLASS_MAPPINGS = {
    "Starnodes_Aspect_Ratio_Advanced": Starnodes_Aspect_Ratio_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Starnodes_Aspect_Ratio_Advanced": "⭐ Starnodes Aspect Ratio Advanced",
}
