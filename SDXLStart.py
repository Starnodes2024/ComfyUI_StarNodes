import os
import json
import torch
import folder_paths
import comfy.sd

class SDXLStartSettings:
    @classmethod
    def INPUT_TYPES(cls):
        # Read ratios
        ratio_sizes, ratio_dict = cls.read_ratios()
        
        # Get available checkpoints
        available_checkpoints = folder_paths.get_filename_list("checkpoints")
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Your positive prompt..."}),
                "Checkpoint": (available_checkpoints, {"tooltip": "The checkpoint (model) to load"}),
                "Latent_Ratio": (ratio_sizes, {"default": "1:1 [1024x1024 square]"}),
                "Latent_Width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Latent_Height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = (
        "MODEL",     # Checkpoint Model
        "CLIP",      # CLIP Model
        "VAE",       # VAE Model
        "LATENT",    # Latent Image
        "INT",       # Width
        "INT",       # Height
        "CONDITIONING"  # Conditioning Output
    )
    
    RETURN_NAMES = (
        "model", 
        "clip", 
        "vae",
        "latent",
        "width", 
        "height",
        "conditioning"  # Conditioning Name
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "StarNodes"

    @staticmethod
    def read_ratios():
        p = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(p, 'sdratios.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
        ratio_sizes = list(data['ratios'].keys())
        ratio_dict = data['ratios']
        
        # User ratios
        user_styles_path = os.path.join(folder_paths.base_path, 'user_ratios.json')
        if os.path.isfile(user_styles_path):
            with open(user_styles_path, 'r') as file:
                user_data = json.load(file)
            for ratio in user_data['ratios']:
                ratio_dict[ratio] = user_data['ratios'][ratio]
                ratio_sizes.append(ratio)
        
        return ratio_sizes, ratio_dict

    @classmethod
    def process_settings(
        cls, 
        text,
        Checkpoint, 
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size
    ):
        # Checkpoint Loading
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", Checkpoint)
        
        # Change this line to capture all returned values
        checkpoint_data = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        
        # Unpack the first 3 values
        model, clip, vae = checkpoint_data[:3]
        
        # Generate Conditioning
        conditioning = None
        if clip and text:
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning = [[cond, output]]
        
        # Latent Image Generation
        _, ratio_dict = cls.read_ratios()
        
        # Explicitly check for Free Ratio
        if Latent_Ratio == "Free Ratio" or "Free" in Latent_Ratio.lower():
            # Use provided width and height
            width = Latent_Width
            height = Latent_Height
        else:
            # Use width and height from the ratio dictionary
            width = ratio_dict[Latent_Ratio]["width"]
            height = ratio_dict[Latent_Ratio]["height"]
        
        # Ensure dimensions are divisible by 8 for latent space
        width = width - (width % 8)
        height = height - (height % 8)
        
        latent = torch.zeros([Batch_Size, 4, height // 8, width // 8])
        
        return (
            model,          # Model
            clip,           # CLIP model
            vae,            # VAE model
            {"samples": latent},  # Latent image
            width,          # Width as an INT output
            height,         # Height as an INT output
            conditioning    # Conditioning
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "SDXLStartSettings": SDXLStartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLStartSettings": "⭐ SD(XL) Star(t) Settings"
}
