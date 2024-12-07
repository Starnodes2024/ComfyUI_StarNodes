import os
import json
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageGrab
import folder_paths
import nodes
import comfy.sd
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel

class Starupscale:
   
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        available_vaes = folder_paths.get_filename_list("vae")
        available_upscalers = folder_paths.get_filename_list("upscale_models")
        
        return {
            "required": {
                "vae_out": (["Default"] + available_vaes, {"default": "ae.safetensors"}),
                "upscale_model": (["Default"] + available_upscalers, {"default": "Default"}),
                "output_longest_side": ("INT", { 
                    "default": 2000, 
                    "min": 0, 
                    "step": 1, 
                    "max": 99999, 
                    "display_name": "Output Size (longest)"
                }),
                "interpolation_mode": (
                    ["bicubic", "bilinear", "nearest", "nearest exact"],
                    {"default": "bicubic"}
                ),
            },
            "optional": {
                "vae_input": ("vae", ),  # Optional VAE input
                "latent_input": ("latent", ),  # Optional latent input
                "image": ("image", ),  # Optional image input
            }
        }
    
    RETURN_TYPES = (
        "vae",       # VAE output
        "image",     # Image output
        "latent",    # Added Latent output
    )
    
    RETURN_NAMES = (
        "output vae",
        "image",
        "latent",  # Added Latent output name
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "StarNodes"
    DESCRIPTION = "TESTNODE FOR NEW FUNCTIONS"

    def process_settings(
        self, 
        vae_out,
        upscale_model,
        output_longest_side,
        interpolation_mode,
        vae_input=None,  # Optional VAE input
        latent_input=None,  # Optional latent input
        image=None,  # Optional image input
    ):
        # VAE Loading
        vaeout = None
        decoder_name = "Default"
        if vae_out != "Default":
            decoder_name = vae_out
            vae = nodes.VAELoader().load_vae(decoder_name)[0]
            vaeout = vae
        
        # Determine processing path
        output_image = None
        
        # Path 1: If both VAE and latent are connected, decode first
        if vae_input is not None and latent_input is not None:
            output_image = nodes.VAEDecode().decode(vae_input, latent_input)[0]
        
        # Path 2: If only image is connected, use the image directly
        elif image is not None:
            output_image = image
        
        # If no image or latent input, create a default black image
        if output_image is None:
            output_image = torch.zeros(1, 64, 64, 3)
        
        # Upscale the image
        if output_image is not None:
            # Load upscale model
            upscale_model = None
            if upscale_model != "Default":
                upscale_model = UpscaleModelLoader().load_model(upscale_model)[0]
            
            # Upscale the image if a model is available
            if upscale_model is not None:
                output_image = ImageUpscaleWithModel().upscale(upscale_model, output_image)[0]
            
            # Resize the image based on longest side
            assert isinstance(output_image, torch.Tensor)
            assert isinstance(output_longest_side, int)
            assert isinstance(interpolation_mode, str)
            
            INTERPOLATION_MODE = interpolation_mode.upper().replace(" ", "_")
            INTERPOLATION_MODE = getattr(InterpolationMode, INTERPOLATION_MODE)
            _, h, w, _ = output_image.shape
            if h >= w:
                new_h = output_longest_side
                new_w = round(w * new_h / h)
            else:  # h < w
                new_w = output_longest_side
                new_h = round(h * new_w / w)
            
            # Resize the image
            output_image = output_image.permute(0, 3, 1, 2)
            output_image = F.resize(
                output_image,
                (new_h, new_w),
                interpolation=INTERPOLATION_MODE,
                antialias=True,
            )
            output_image = output_image.permute(0, 2, 3, 1)

        # Encode the resized image to latent representation ALWAYS using the loaded VAE
        output_latent = None
        if vaeout is not None and output_image is not None:
            # Use the loaded VAE to encode the output image to latent space
            output_latent = nodes.VAEEncode().encode(vaeout, output_image)[0]

        return (
            vaeout,           # VAE output
            output_image,     # Image output
            output_latent,    # Added Latent output
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "Starupscale": Starupscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Starupscale": "⭐ Star Model Latent Upscaler"
}