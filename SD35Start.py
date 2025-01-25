import os
import json
import torch
import nodes
import folder_paths
import comfy.sd

class SD35StartSettings:
    @classmethod
    def INPUT_TYPES(cls):
        # Pfade und Modelle laden
        models_paths, _ = folder_paths.folder_names_and_paths.get(
            "unet", folder_paths.folder_names_and_paths.get("unet", [[], set()])
        )
        
        available_models = ["Default"]
        available_unets = folder_paths.get_filename_list("unet")
        available_clips = folder_paths.get_filename_list("text_encoders")
        available_vaes = folder_paths.get_filename_list("vae")
        
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
        
        # Ratio-Einstellungen laden
        ratio_sizes, ratio_dict = cls.read_ratios()
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Your positive prompt..."}),
                "UNET": (["Default"] + available_unets, {"default": "sd3.5_large_turbo.safetensors"}),
                "CLIP_1": (["Default"] + available_clips, {"default": "clip_l.safetensors"}),
                "CLIP_2": (["Default"] + available_clips, {"default": "clip_g.safetensors"}),
                "CLIP_3": (["Default"] + available_clips, {"default": "t5xxl_fp16.safetensors"}),
                "VAE": (["Default"] + available_vaes, {"default": "stableDiffusion35VAE_official.safetensors"}),
                "Weight_Dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "Latent_Ratio": (ratio_sizes, {"default": "1:1 [1024x1024 square]"}),
                "Latent_Width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Latent_Height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = (
        "MODEL",     # UNET Model
        "CLIP",      # CLIP Model
        "LATENT",    # Latent Image
        "INT",       # Width
        "INT",       # Height
        "CONDITIONING",  # Conditioning hinzugefügt
        "VAE"       # Added VAE output
    )
    
    RETURN_NAMES = (
        "UNET", 
        "CLIP", 
        "LATENT",
        "WIDTH",
        "HEIGHT",
        "CONDITIONING",  
        "VAE"
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "⭐StarNodes"

    @staticmethod
    def read_ratios():
        p = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(p, 'sd3ratios.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
        ratio_sizes = list(data['ratios'].keys())
        ratio_dict = data['ratios']
        
        # Benutzerdefinierte Ratios
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
        CLIP_3, 
        VAE,
        Weight_Dtype, 
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size
    ):
        # UNET laden
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

            unet_path = folder_paths.get_full_path_or_raise("unet", UNET)
            unet = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        
        # CLIP laden und Conditioning erzeugen
        conditioning = None
        clip = None
        if CLIP_1 != "Default" and CLIP_2 != "Default" and CLIP_3 != "Default":
            clip_paths = [
                folder_paths.get_full_path_or_raise("text_encoders", CLIP_1),
                folder_paths.get_full_path_or_raise("text_encoders", CLIP_2),
                folder_paths.get_full_path_or_raise("text_encoders", CLIP_3)
            ]
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.SD3
            )
            
            if clip:
                tokens = clip.tokenize(text)
                output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = output.pop("cond")
                conditioning = [[cond, output]]
                
        # VAE Loading
        vae = None
        decoder_name = "Default"
        if VAE != "Default":
            decoder_name = VAE
            vae = nodes.VAELoader().load_vae(decoder_name)[0]

        # Latentbild generieren
        _, ratio_dict = self.read_ratios()
        if Latent_Ratio == "Free Ratio" or "Free" in Latent_Ratio.lower():
            width, height = Latent_Width, Latent_Height
        else:
            width, height = ratio_dict[Latent_Ratio]["width"], ratio_dict[Latent_Ratio]["height"]
        
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([Batch_Size, 4, height // 8, width // 8])
        
        return (
            unet,
            clip,
            {"samples": latent},
            width,
            height,
            conditioning,  # Conditioning hinzufügen
            vae            # Added VAE output
        )

# Node-Mapping für ComfyUI
NODE_CLASS_MAPPINGS = {
    "SD35StartSettings": SD35StartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD35StartSettings": "⭐ SD3.5 Star(t) Settings"
}
