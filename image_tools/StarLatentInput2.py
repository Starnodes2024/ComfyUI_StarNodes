import torch

class StarLatentSwitch2:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_out",)
    FUNCTION = "process_latents"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "Latent 1": ("LATENT",),
                "Latent 2": ("LATENT",),
                "Latent 3": ("LATENT",),
                "Latent 4": ("LATENT",),
                "Latent 5": ("LATENT",),
            }
        }

    def process_latents(self, **kwargs):
        # Return the first connected latent
        for i in range(1, 6):
            latent = kwargs.get(f"Latent {i}")
            if latent is not None:
                return (latent,)
        # If no latent is connected, create a default empty latent
        batch_size = 1
        height = 64  # 512 // 8
        width = 64   # 512 // 8
        channels = 4
        default_latent = {
            "samples": torch.zeros((batch_size, channels, height, width)),
        }
        return (default_latent,)

NODE_CLASS_MAPPINGS = dict()
NODE_CLASS_MAPPINGS["StarLatentSwitch2"] = StarLatentSwitch2

NODE_DISPLAY_NAME_MAPPINGS = dict()
NODE_DISPLAY_NAME_MAPPINGS["StarLatentSwitch2"] = "⭐ Star Latent Input 2 (Optimized)"
