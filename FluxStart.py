import os
import json
import random
import torch
import folder_paths
import nodes
import comfy.sd
import comfy.utils
import types

class FluxStartSettings:
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
        
        # Set default VAE device to first CUDA device if available
        default_vae_device = cuda_devices[0] if cuda_devices else "cpu"

        # Existing model path and model loading logic
        models_paths, _ = folder_paths.folder_names_and_paths.get("diffusion_models", 
                          folder_paths.folder_names_and_paths.get("unet", [[], set()]))
        
        available_models = ["Default"]
        available_unets = folder_paths.get_filename_list("diffusion_models")
        available_clips = folder_paths.get_filename_list("text_encoders")
        available_vaes = ["Default"] + cls.vae_list()
        
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
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "", "placeholder": "Your positive prompt..."}),
                "UNET": (["Default"] + available_unets, {"default": "flux1-dev.safetensors"}),
                "CLIP_1": (["Default"] + available_clips, {"default": "t5xxl_fp16.safetensors"}),
                "CLIP_2": (["Default"] + available_clips, {"default": "ViT-L-14-BEST-smooth-GmP-ft.safetensors"}),
                "CLIP_Device": (devices, {"default": "cpu"}),
                "VAE": (available_vaes, {"default": "ae.safetensors"}),
                "VAE_Device": (devices, {"default": default_vae_device}),
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
        "CONDITIONING",  # Added conditioning output
        "VAE",       # Added VAE output
    )
    
    RETURN_NAMES = (
        "UNET", 
        "CLIP", 
        "LATENT",
        "WIDTH",
        "HEIGHT",
        "CONDITIONING",  # Added conditioning output name
        "VAE",
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "⭐StarNodes"
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

    def override_device(self, model, model_attr, device):
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

    def process_settings(
        self, 
        text,
        UNET, 
        CLIP_1, 
        CLIP_2,
        CLIP_Device,
        VAE,
        VAE_Device,
        Weight_Dtype, 
        Latent_Ratio,
        Latent_Width,
        Latent_Height,
        Batch_Size
    ):
        # If no text input is provided, use default creative prompt
        if not text.strip():
            text = "a confused looking fluffy purple monster with a \"?\" sign"
            
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
            
            # Set CLIP device
            if clip is not None:
                clip_device = torch.device(CLIP_Device)
                clip = self.override_device(clip, "cond_stage_model", clip_device)
                
                # Generate conditioning using both CLIPs
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

        # VAE Loading
        vae = None
        if VAE != "Default":
            if VAE in ["taesd", "taesdxl", "taesd3", "taef1"]:
                sd = self.load_taesd(VAE)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", VAE)
                sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            
            # Set VAE device
            if vae is not None:
                vae_device = torch.device(VAE_Device)
                vae = self.override_device(vae, "first_stage_model", vae_device)

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
            conditioning,   # Added conditioning output
            vae,            # Added VAE output
        )

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FluxStartSettings": FluxStartSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxStartSettings": "⭐ FLUX Star(t) Settings"
}