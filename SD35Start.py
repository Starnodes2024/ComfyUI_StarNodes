import os
import json
import random
import torch
import folder_paths
import nodes
import comfy.sd

class SD35StartSettings:
    @classmethod
    def INPUT_TYPES(cls):
        # Existing model path and model loading logic
        models_paths, _ = folder_paths.folder_names_and_paths.get("unet", 
                          folder_paths.folder_names_and_paths.get("unet", [[], set()]))
        
        available_models = ["Default"]
        available_unets = folder_paths.get_filename_list("unet")
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
                "UNET": (["Default"] + available_unets, {"default": "sd3.5_large_turbo.safetensors"}),
                "CLIP_1": (["Default"] + available_clips, {"default": "clip_l.safetensors"}),
                "CLIP_2": (["Default"] + available_clips, {"default": "clip_g.safetensors"}),
                "CLIP_3": (["Default"] + available_clips, {"default": "t5xxl_fp16.safetensors"}),
                "Weight_Dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "Latent_Ratio": (ratio_sizes, {"default": "1:1 [1024x1024 square]"}),
                "Latent_Width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "Latent_Height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = (
        "MODEL",     # UNET Model
        "CLIP",      # CLIP Model
        "LATENT",    # Latent Image
        "INT",       # Width
        "INT"        # Height
    )
    RETURN_NAMES = (
        "UNET", 
        "CLIP", 
        "LATENT",
        "WIDTH",
        "HEIGHT"
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "StarNodes"

    @staticmethod
    def read_ratios():
        p = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(p, 'sd3ratios.json')
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
        UNET, 
        CLIP_1, 
        CLIP_2, 
        CLIP_3, 
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
        
        # CLIP Loading
        clip = None
        if CLIP_1 != "Default" and CLIP_2 != "Default" and CLIP_3 != "Default":
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_2)
            clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_3)
            clip_type = comfy.sd.CLIPType.FLUX
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], 
                                      embedding_directory=folder_paths.get_folder_paths("embeddings"), 
                                      clip_type=clip_type)
        
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
        
        latent = torch.zeros([Batch_Size, 16, height // 8, width // 8])
        
        return (
            unet,           # UNET model or None
            clip,           # CLIP model or None
            {"samples": latent},  # Latent image
            width,          # Width as an INT output
            height          # Height as an INT output
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "SD35StartSettings": SD35StartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD35StartSettings": "⭐ SD3.5 Star(t) Settings"
}