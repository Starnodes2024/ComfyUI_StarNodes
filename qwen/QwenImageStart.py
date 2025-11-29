import os
import json
import torch
import nodes
import folder_paths
import comfy.sd
import comfy.utils
import types

class QwenImageStartSettings:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(cls):
        # Get available devices
        devices = ["cpu"]
        cuda_devices = [f"cuda:{k}" for k in range(0, torch.cuda.device_count())]
        devices.extend(cuda_devices)

        # Get available models
        available_diffusion_models = folder_paths.get_filename_list("diffusion_models")
        available_unets = folder_paths.get_filename_list("unet")
        available_clips = folder_paths.get_filename_list("clip")
        available_vaes = ["Default"] + cls.vae_list()
        
        # Combine diffusion_models and unet for Diffusion_Model selector
        diffusion_model_list = ["Default"] + available_diffusion_models + available_unets
        
        # Read Qwen ratios
        ratio_sizes, ratio_dict = cls.read_qwen_ratios()
        
        return {
            "required": {
                "Positive_Prompt": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "placeholder": "Your positive prompt...",
                    "tooltip": "Positive prompt for image generation. If empty, generates a fluffy purple monster."
                }),
                "Negative_Prompt": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "placeholder": "Your negative prompt...",
                    "tooltip": "Negative prompt for image generation. If empty, creates a zero-out condition."
                }),
                "Diffusion_Model": (diffusion_model_list, {
                    "default": "Default",
                    "tooltip": "Select diffusion model from models/diffusion_models or models/unet folders"
                }),
                "VAE": (available_vaes, {
                    "default": "Default",
                    "tooltip": "Select VAE model from models/vae folder"
                }),
                "CLIP": (["Default"] + available_clips, {
                    "default": "Default",
                    "tooltip": "Select CLIP model from models/clip folder"
                }),
                "CLIP_Type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "flux", "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], {
                    "default": "qwen_image",
                    "tooltip": "CLIP model type for loading. For Qwen models, use 'qwen_image'"
                }),
                "CLIP_Device": (devices, {
                    "default": "cpu",
                    "tooltip": "Device to load CLIP model on"
                }),
                "Latent_Ratio": (ratio_sizes, {
                    "default": "1:1 (1328x1328)",
                    "tooltip": "Predefined aspect ratios for Qwen Image models"
                }),
                "Latent_Width": ("INT", {
                    "default": 1328, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 16,
                    "tooltip": "Custom width when using Free Ratio"
                }),
                "Latent_Height": ("INT", {
                    "default": 1328, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 16,
                    "tooltip": "Custom height when using Free Ratio"
                }),
                "Batch_Size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4096,
                    "tooltip": "Number of images to generate in batch"
                }),
                "use_nearest_image_ratio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled with image input, automatically selects the closest aspect ratio"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for automatic aspect ratio detection"
                }),
                "model_override": ("MODEL", {
                    "tooltip": "Optional model input. When connected, bypasses the Diffusion_Model selector and uses this model directly."
                }),
                "LoRA_Stack": ("LORA_STACK", {
                    "tooltip": "Optional stack of LoRAs to apply to the model and internal conditioning."
                }),
            }
        }
    
    RETURN_TYPES = (
        "MODEL",         # model
        "CLIP",          # clip
        "VAE",           # vae
        "LATENT",        # latent
        "INT",           # width
        "INT",           # height
        "CONDITIONING",  # condition_pos
        "CONDITIONING",  # condition_neg
        "STRING",        # prompt_pos
        "STRING",        # prompt_neg
    )
    
    RETURN_NAMES = (
        "model", 
        "clip", 
        "vae",
        "latent",
        "width",
        "height",
        "condition_pos",
        "condition_neg",
        "prompt_pos",
        "prompt_neg",
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "⭐StarNodes/Starters"
    DESCRIPTION = "Qwen Image Star(t) Settings - All-in-one starter node for Qwen diffusion models"

    @staticmethod
    def read_qwen_ratios():
        """Read Qwen-specific ratios"""
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
        ratio_sizes = list(ratios.keys())
        return ratio_sizes, ratios

    def override_device(self, model, model_attr, device):
        """Override device for model"""
        # Set model/patcher attributes
        model.device = device
        patcher = getattr(model, "patcher", model)
        for name in ["device", "load_device", "offload_device", "current_device", "output_device"]:
            setattr(patcher, name, device)

        # Move model to device
        py_model = getattr(model, model_attr)
        py_model.to = types.MethodType(torch.nn.Module.to, py_model)
        py_model.to(device)

        # Remove ability to move model
        def to(*args, **kwargs):
            pass
        py_model.to = types.MethodType(to, py_model)
        return model

    def find_nearest_ratio(self, image, ratio_dict):
        """Find the nearest aspect ratio from available ratios based on input image"""
        try:
            # image is [B,H,W,C] or [H,W,C]
            if image.ndim == 3:
                H, W = int(image.shape[0]), int(image.shape[1])
            else:
                H, W = int(image.shape[1]), int(image.shape[2])
            
            if H > 0 and W > 0:
                target_ar = W / H
                candidates = [(k, v) for k, v in ratio_dict.items() if v is not None]
                
                best = None
                best_err = 1e9
                for label, (w, h) in candidates:
                    ar = w / h
                    err = abs(ar - target_ar)
                    if err < best_err:
                        best_err = err
                        best = (w, h)
                
                if best is not None:
                    return best
        except Exception:
            pass
        return None

    def process_settings(
        self, 
        Positive_Prompt,
        Negative_Prompt,
        Diffusion_Model,
        VAE,
        CLIP,
        CLIP_Type,
        CLIP_Device,
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size,
        use_nearest_image_ratio=False,
        image=None,
        model_override=None,
        LoRA_Stack=None
    ):
        # Default prompts when input is empty
        if not Positive_Prompt.strip():
            Positive_Prompt = "a confused looking fluffy purple monster with a \"?\" sign"
        
        # Store original prompts for string outputs
        prompt_pos = Positive_Prompt
        prompt_neg = Negative_Prompt
        
        # Diffusion Model Loading
        model = None
        
        # Check if model_override is provided
        if model_override is not None:
            print("[QwenImageStart] Using model_override input, ignoring Diffusion_Model selector")
            model = model_override
        elif Diffusion_Model != "Default":
            # Try diffusion_models first, then unet
            try:
                model_path = folder_paths.get_full_path_or_raise("diffusion_models", Diffusion_Model)
                model = comfy.sd.load_diffusion_model(model_path)
            except:
                try:
                    model_path = folder_paths.get_full_path_or_raise("unet", Diffusion_Model)
                    model = comfy.sd.load_diffusion_model(model_path)
                except Exception as e:
                    print(f"Error loading diffusion model: {e}")
        
        # CLIP Loading
        clip = None
        if CLIP != "Default":
            try:
                clip_path = folder_paths.get_full_path_or_raise("clip", CLIP)
                
                # Map CLIP_Type string to CLIPType enum
                clip_type_map = {
                    "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
                    "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,
                    "sd3": comfy.sd.CLIPType.SD3,
                    "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,
                    "mochi": comfy.sd.CLIPType.MOCHI,
                    "ltxv": comfy.sd.CLIPType.LTXV,
                    "flux": comfy.sd.CLIPType.FLUX,
                    "hunyuan_video": comfy.sd.CLIPType.HUNYUAN_VIDEO,
                    "pixart": comfy.sd.CLIPType.PIXART,
                    "cosmos": comfy.sd.CLIPType.COSMOS,
                    "lumina2": comfy.sd.CLIPType.LUMINA2,
                    "wan": comfy.sd.CLIPType.WAN,
                    "hidream": comfy.sd.CLIPType.HIDREAM,
                    "chroma": comfy.sd.CLIPType.CHROMA,
                    "ace": comfy.sd.CLIPType.ACE,
                    "omnigen2": comfy.sd.CLIPType.OMNIGEN2,
                    "qwen_image": comfy.sd.CLIPType.QWEN_IMAGE,
                    "hunyuan_image": comfy.sd.CLIPType.HUNYUAN_IMAGE,
                }
                
                clip_type_enum = clip_type_map.get(CLIP_Type, comfy.sd.CLIPType.STABLE_DIFFUSION)
                
                clip = comfy.sd.load_clip(
                    ckpt_paths=[clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type_enum
                )
                
                # Set CLIP device
                if clip is not None:
                    clip_device = torch.device(CLIP_Device)
                    clip = self.override_device(clip, "cond_stage_model", clip_device)
            except Exception as e:
                print(f"Error loading CLIP: {e}")
        
        # Generate Conditioning
        conditioning_pos = None
        conditioning_neg = None
        
        # Create a copy of clip for conditioning so we can apply LoRAs independently
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
        
        if clip_for_cond is not None:
            # Positive conditioning
            if Positive_Prompt.strip():
                tokens = clip_for_cond.tokenize(Positive_Prompt)
                output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = output.pop("cond")
                conditioning_pos = [[cond, output]]
            
            # Negative conditioning - create zero-out condition if empty
            if Negative_Prompt.strip():
                tokens = clip_for_cond.tokenize(Negative_Prompt)
                output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = output.pop("cond")
                conditioning_neg = [[cond, output]]
            else:
                # Create empty/zero conditioning for negative
                tokens = clip_for_cond.tokenize("")
                output = clip_for_cond.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                cond = output.pop("cond")
                conditioning_neg = [[cond, output]]
        
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
        
        # VAE Loading
        vae = None
        if VAE != "Default":
            if VAE in ["taesd", "taesdxl", "taesd3", "taef1"]:
                sd = self.load_taesd(VAE)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", VAE)
                sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
        
        # Latent Image Generation with aspect ratio handling
        _, ratio_dict = self.read_qwen_ratios()
        
        # Check if we should use nearest image ratio
        if use_nearest_image_ratio and image is not None:
            nearest = self.find_nearest_ratio(image, ratio_dict)
            if nearest is not None:
                width, height = nearest
            else:
                # Fallback to selected ratio
                if Latent_Ratio == "Free Ratio (custom)" or "Free" in Latent_Ratio:
                    width, height = Latent_Width, Latent_Height
                else:
                    width, height = ratio_dict[Latent_Ratio]
        else:
            # Use selected ratio
            if Latent_Ratio == "Free Ratio (custom)" or "Free" in Latent_Ratio:
                width, height = Latent_Width, Latent_Height
            else:
                width, height = ratio_dict[Latent_Ratio]
        
        # Ensure dimensions are divisible by 16
        width = width - (width % 16)
        height = height - (height % 16)
        
        # Ensure divisibility by 8 for latent space
        width = width - (width % 8)
        height = height - (height % 8)
        
        latent = torch.zeros([Batch_Size, 4, height // 8, width // 8])
        
        return (
            model,
            clip,
            vae,
            {"samples": latent},
            width,
            height,
            conditioning_pos,
            conditioning_neg,
            prompt_pos,
            prompt_neg,
        )

# Node-Mapping für ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageStartSettings": QwenImageStartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageStartSettings": "⭐ Qwen Image Star(t) Settings"
}
