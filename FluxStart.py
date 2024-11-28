import os
import json
import random
import torch
import folder_paths
import nodes
import comfy.sd

class FluxStartSettings:
    @classmethod
    def INPUT_TYPES(cls):
        # Existing model path and model loading logic
        models_paths, _ = folder_paths.folder_names_and_paths.get("diffusion_models", 
                          folder_paths.folder_names_and_paths.get("unet", [[], set()]))
        
        available_models = ["Default"]
        available_unets = folder_paths.get_filename_list("diffusion_models")
        available_clips = folder_paths.get_filename_list("text_encoders")
        
        try:
            for path in models_paths:
                if os.path.exists(path):
                    available_models.extend([
                        f for f in os.listdir(path) 
                        if os.path.isfile(os.path.join(path, f)) and 
                        (f.endswith('.safetensors') or f.endswith('.ckpt') or f.endswith('.pt'))
                    ])
        except Exception as e:
            print(f"Error reading diffusion models folder: {e}")
        
        # Read ratios
        ratio_sizes, ratio_dict = cls.read_ratios()
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Your positive prompt..."}),
                "UNET": (["Default"] + available_unets, {"default": "flux1-dev.safetensors"}),
                "CLIP_1": (["Default"] + available_clips, {"default": "t5xxl_fp16.safetensors"}),
                "CLIP_2": (["Default"] + available_clips, {"default": "ViT-L-14-BEST-smooth-GmP-ft.safetensors"}),
                "Weight_Dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "Latent_Ratio": (ratio_sizes, {"default": "1:1 [1024x1024 square]"}),
                "Latent_Width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "Latent_Height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    
    RETURN_TYPES = (
        "MODEL",     # UNET Model
        "CLIP",      # CLIP Model
        "LATENT",    # Latent Image
        "INT",       # Width
        "INT",       # Height
        "CONDITIONING"  # Added conditioning output
    )
    
    RETURN_NAMES = (
        "UNET", 
        "CLIP", 
        "LATENT",
        "WIDTH",
        "HEIGHT",
        "CONDITIONING"  # Added conditioning output name
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "StarNodes"
    DESCRIPTION = "Flux Start Settings with text conditioning"

    @staticmethod
    def read_ratios():
        p = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(p, 'fluxratios.json')
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

    def process_settings(
        self, 
        text,
        UNET, 
        CLIP_1, 
        CLIP_2, 
        Weight_Dtype, 
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size
    ):
        # UNET Loading
        unet = None
        if UNET != "Default":
            model_options = {}
            if Weight_Dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif Weight_Dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif Weight_Dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", UNET)
            unet = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        
        # CLIP Loading and Conditioning
        conditioning = None
        clip = None
        if CLIP_1 != "Default" and CLIP_2 != "Default":
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_2)
            
            # Ensure we're using the Flux-specific CLIP type
            clip_type = comfy.sd.CLIPType.FLUX
            
            # Load both CLIP models
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2], 
                embedding_directory=folder_paths.get_folder_paths("embeddings"), 
                clip_type=clip_type
            )
            
            # Generate conditioning using both CLIPs
            if clip is not None:
                # Tokenize the text
                tokens = clip.tokenize(text)
                
                # Encode tokens using both CLIPs
                output = clip.encode_from_tokens(
                    tokens, 
                    return_pooled=True, 
                    return_dict=True
                )
                
                # Extract the conditioning
                cond = output.pop("cond")
                conditioning = [[cond, output]]

        # Latent Image Generation
        _, ratio_dict = self.read_ratios()
        
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
            unet,           # UNET model or None
            clip,           # CLIP model or None
            {"samples": latent},  # Latent image
            width,          # Width as an INT output
            height,         # Height as an INT output
            conditioning    # Added conditioning output
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FluxStartSettings": FluxStartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxStartSettings": "⭐ Flux Star(t) Settings"
}