import torch

class StarQwenImageRatio:
    @classmethod
    def INPUT_TYPES(cls):
        ratios = {
            "1:1 (1328x1328)": (1328, 1328),
            "16:9 (1664x928)": (1664, 928),
            "9:16 (928x1664)": (928, 1664),
            "4:3 (1472x1104)": (1472, 1104),
            "3:4 (1104x1472)": (1104, 1472),
            "3:2 (1584x1056)": (1584, 1056),
            "2:3 (1056x1584)": (1056, 1584),
            "5:7 (1120x1568)": (1120, 1568),
            "7:5 (1568x1120)": (1568, 1120),
            "Free Ratio (custom)": None,
        }
        return {
            "required": {
                "ratio": (list(ratios.keys()), {"default": "1:1 (1328x1328)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "custom_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def create(self, ratio, batch_size=1, custom_width=1024, custom_height=1024):
        mapping = {
            "1:1 (1328x1328)": (1328, 1328),
            "16:9 (1664x928)": (1664, 928),
            "9:16 (928x1664)": (928, 1664),
            "4:3 (1472x1104)": (1472, 1104),
            "3:4 (1104x1472)": (1104, 1472),
            "3:2 (1584x1056)": (1584, 1056),
            "2:3 (1056x1584)": (1056, 1584),
            "5:7 (1120x1568)": (1120, 1568),
            "7:5 (1568x1120)": (1568, 1120),
        }
        if "Free" in ratio:
            width, height = custom_width, custom_height
        else:
            width, height = mapping[ratio]

        # Enforce divisibility by 16 as requested
        width = width - (width % 16)
        height = height - (height % 16)

        # Ensure divisibility by 8 for latent grid (SD3.5 uses 8x downsample like SDXL in this repo)
        width_latent = width - (width % 8)
        height_latent = height - (height % 8)

        latent = torch.zeros([batch_size, 4, height_latent // 8, width_latent // 8])
        return ({"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "StarQwenImageRatio": StarQwenImageRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenImageRatio": "⭐ Star Qwen Image Ratio",
}
