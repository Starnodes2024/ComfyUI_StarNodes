import torch

class StarLatentSwitch:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    CATEGORY = '⭐StarNodes'
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_out",)
    FUNCTION = "process_latents"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "Latent 1": ("LATENT",),
            }
        }

    def process_latents(self, **kwargs):
        # Try to get the first connected latent
        for i in range(1, 21):  # Support up to 20 inputs
            latent = kwargs.get(f"Latent {i}")
            if latent is not None:
                return (latent,)
        
        # If no latent is connected, create a default empty latent
        # Default size similar to 512x512 image latent representation
        batch_size = 1
        height = 64  # 512 // 8
        width = 64   # 512 // 8
        channels = 4
        
        default_latent = {
            "samples": torch.zeros((batch_size, channels, height, width)),
        }
        
        return (default_latent,)

NODE_CLASS_MAPPINGS = {
    "StarLatentSwitch": StarLatentSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLatentSwitch": "⭐ Star Latent Input (Dynamic)"
}
