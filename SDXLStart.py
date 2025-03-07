import os
import json
import torch
import folder_paths
import comfy.sd

class SDXLStartSettings:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    @classmethod
    def INPUT_TYPES(cls):
        # Read ratios
        ratio_sizes, ratio_dict = cls.read_ratios()
        
        # Get available checkpoints
        available_checkpoints = folder_paths.get_filename_list("checkpoints")
        available_vaes = folder_paths.get_filename_list("vae")
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "Your positive prompt..."}),
                "negative_text": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "Your negative prompt..."}),
                "Checkpoint": (available_checkpoints, {"tooltip": "The checkpoint (model) to load"}),
                "VAE": (["Default"] + available_vaes, {"default": "Default"}),
                "Latent_Ratio": (ratio_sizes, {"default": "1:1 [1024x1024 square]"}),
                "Latent_Width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Latent_Height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "LoRA_Stack": ("LORA_STACK", {"tooltip": "Optional stack of LoRAs to apply to the model and internal conditioning."})
            }
        }
    
    RETURN_TYPES = (
        "MODEL",     # Checkpoint Model
        "CLIP",      # CLIP Model
        "VAE",       # VAE Model
        "LATENT",    # Latent Image
        "INT",       # Width
        "INT",       # Height
        "CONDITIONING",  # Positive Conditioning Output
        "CONDITIONING"   # Negative Conditioning Output
    )
    
    RETURN_NAMES = (
        "model", 
        "clip", 
        "vae",
        "latent",
        "width", 
        "height",
        "conditioning_POS",  # Positive Conditioning Name
        "conditioning_NEG"   # Negative Conditioning Name
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "⭐StarNodes"

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
        negative_text,
        Checkpoint,
        VAE,
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size,
        LoRA_Stack=None
    ):
        # Default prompts when input is empty
        if not text.strip():
            text = "a confused looking fluffy purple monster with a \"?\" sign"
        if not negative_text.strip():
            negative_text = "bad quality"

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
        
        # Load custom VAE if specified
        if VAE != "Default":
            vae_path = folder_paths.get_full_path_or_raise("vae", VAE)
            vae = comfy.sd.VAE(ckpt_path=vae_path)
        
        # Create a copy of clip for conditioning
        clip_for_cond = clip.clone() if clip is not None else None
        
        # Apply LoRAs to the conditioning CLIP if provided
        if LoRA_Stack is not None and len(LoRA_Stack) > 0 and clip_for_cond is not None:
            for lora_item in LoRA_Stack:
                lora_name = lora_item["name"]
                clip_strength = lora_item["clip_strength"]
                lora = lora_item["lora"]
                
                # Apply LoRA to conditioning CLIP
                if clip_strength != 0:
                    _, clip_for_cond = comfy.sd.load_lora_for_models(
                        None, 
                        clip_for_cond, 
                        lora, 
                        0, 
                        clip_strength
                    )
        
        # Generate Positive Conditioning
        conditioning_pos = None
        if clip_for_cond and text:
            tokens = clip_for_cond.tokenize(text)
            output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning_pos = [[cond, output]]
        
        # Generate Negative Conditioning
        conditioning_neg = None
        if clip_for_cond and negative_text:
            tokens = clip_for_cond.tokenize(negative_text)
            output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conditioning_neg = [[cond, output]]
        
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
        
        # Apply LoRAs from stack to model and clip if provided
        if LoRA_Stack is not None and len(LoRA_Stack) > 0 and model is not None:
            for lora_item in LoRA_Stack:
                lora_name = lora_item["name"]
                model_strength = lora_item["model_strength"]
                clip_strength = lora_item["clip_strength"]
                lora = lora_item["lora"]
                
                # Apply LoRA to model and clip
                if model_strength != 0:
                    model, clip = comfy.sd.load_lora_for_models(
                        model, 
                        clip, 
                        lora, 
                        model_strength, 
                        clip_strength if clip is not None else 0
                    )
        
        return (
            model, 
            clip, 
            vae, 
            {"samples": latent},  # Latent image
            width, 
            height,
            conditioning_pos,
            conditioning_neg
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "SDXLStartSettings": SDXLStartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLStartSettings": "⭐ SD(XL) Star(t) Settings"
}
