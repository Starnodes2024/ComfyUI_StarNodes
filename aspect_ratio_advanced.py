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

        megapixel_choices = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        megapixel_choices_str = [str(mp) for mp in megapixel_choices]

        return {
            "optional": {
                "image": ("IMAGE",),
            },
            "required": {
                "aspect_ratio": (ratio_choices, ),
                "megapixel": (megapixel_choices_str, {"default": "1.0"}),
                "use_nearest_from_image": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("width", "height", "Resolution", "SDXL / FLUX", "SD3.5")
    FUNCTION = "get_resolution"
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def get_resolution(self, aspect_ratio, megapixel, use_nearest_from_image=False, image=None, batch_size=1):
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
        rw, rh = get_ratio_dims(selected_ratio_key)
        mp = float(megapixel)
        total_pixels = mp * 1_000_000
        width = int(round((total_pixels * rw / rh) ** 0.5))
        height = int(round(width * rh / rw))
        resolution_str = f"{width} x {height}"

        # Latent for Flux/SDXL: shape [batch, 4, H//8, W//8]
        width_latent = width - (width % 8)
        height_latent = height - (height % 8)
        latent_flux_sdxl = torch.zeros([batch_size, 4, height_latent // 8, width_latent // 8])

        # Latent for SD3.5: shape [batch, 4, H//16, W//16]
        width_latent_sd35 = width - (width % 16)
        height_latent_sd35 = height - (height % 16)
        latent_sd35 = torch.zeros([batch_size, 4, height_latent_sd35 // 16, width_latent_sd35 // 16])

        return (width, height, resolution_str, {"samples": latent_flux_sdxl}, {"samples": latent_sd35})

NODE_CLASS_MAPPINGS = {
    "Starnodes_Aspect_Ratio_Advanced": Starnodes_Aspect_Ratio_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Starnodes_Aspect_Ratio_Advanced": "⭐ Starnodes Aspect Ratio Advanced",
}
