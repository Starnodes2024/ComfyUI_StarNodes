import torch

class Star_Image2Latent:
    """
    Converts an input image to a latent, adds configurable noise, and outputs the latent.
    Supports SD, SDXL, SD3.5, and FLUX latent types.
    """
    CATEGORY = "⭐StarNodes/Image And Latent"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "noise_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "latent_type": (["SD", "SDXL", "SD3.5", "FLUX"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "image2latent"
    OUTPUT_NODE = False

    def image2latent(self, image, vae, noise_amount, latent_type):
        # Encode image to latent using VAE
        latent = vae.encode(image)
        # Ensure latent is a torch tensor
        if not isinstance(latent, torch.Tensor):
            latent = torch.from_numpy(latent)
        device = latent.device
        dtype = latent.dtype

        # Model channel expectations
        expected_channels = {
            "SD": 4,
            "SDXL": 4,
            "SD3.5": 4,
            "FLUX": 8,
        }
        exp_channels = expected_channels.get(latent_type, latent.shape[1])
        if latent.shape[1] != exp_channels:
            print(f"[Star_Image2Latent] WARNING: Latent has {latent.shape[1]} channels but model '{latent_type}' expects {exp_channels} channels. Please check your VAE and model compatibility.")

        # Add noise using torch
        noise = torch.randn_like(latent)
        noised_latent = (1.0 - noise_amount) * latent + noise_amount * noise

        # Return as dict for ComfyUI
        return ({"samples": noised_latent},)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Star_Image2Latent": Star_Image2Latent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Star_Image2Latent": "⭐ Star Image2Latent",
}
