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
import comfy.utils
import types
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel

class Starupscale:
   
    def __init__(self):
        pass
        
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
        
        available_vaes = ["Default"] + cls.vae_list()
        available_upscalers = folder_paths.get_filename_list("upscale_models")
        
        return {
            "required": {
                "VAE_OUT": (available_vaes, {"default": "ae.safetensors"}),
                "VAE_Device": (devices, {"default": default_vae_device}),
                "UPSCALE_MODEL": (["Default"] + available_upscalers, {"default": "Default"}),
                "OUTPUT_LONGEST_SIDE": ("INT", { 
                    "default": 2000, 
                    "min": 0, 
                    "step": 1, 
                    "max": 99999, 
                    "display_name": "Output Size (longest)"
                }),
                "INTERPOLATION_MODE": (
                    ["bicubic", "bilinear", "nearest", "nearest exact"],
                    {"default": "bicubic"}
                ),
            },
            "optional": {
                "VAE_INPUT": ("VAE", ),  # Optional VAE input
                "LATENT_INPUT": ("LATENT", ),  # Optional latent input
                "IMAGE": ("IMAGE", ),  # Optional image input
            }
        }
    
    RETURN_TYPES = (
        "VAE",       # VAE output
        "IMAGE",     # Image output
        "LATENT",    # Added Latent output
    )
    
    RETURN_NAMES = (
        "OUTPUT VAE",
        "IMAGE",
        "LATENT",  # Added Latent output name
    )
    
    FUNCTION = "process_settings"
    CATEGORY = "⭐StarNodes"
    DESCRIPTION = "TESTNODE FOR NEW FUNCTIONS"

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
        VAE_OUT,
        VAE_Device,
        UPSCALE_MODEL,
        OUTPUT_LONGEST_SIDE,
        INTERPOLATION_MODE,
        VAE_INPUT=None,  # Optional VAE input
        LATENT_INPUT=None,  # Optional latent input
        IMAGE=None,  # Optional image input
    ):
        # VAE Loading
        vaeout = None
        if VAE_OUT != "Default":
            if VAE_OUT in ["taesd", "taesdxl", "taesd3", "taef1"]:
                sd = self.load_taesd(VAE_OUT)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", VAE_OUT)
                sd = comfy.utils.load_torch_file(vae_path)
            vaeout = comfy.sd.VAE(sd=sd)
            
            # Set VAE device
            if vaeout is not None:
                vae_device = torch.device(VAE_Device)
                vaeout = self.override_device(vaeout, "first_stage_model", vae_device)
        
        # Determine processing path
        output_image = None
        
        # Path 1: If both VAE and latent are connected, decode first
        if VAE_INPUT is not None and LATENT_INPUT is not None:
            output_image = nodes.VAEDecode().decode(VAE_INPUT, LATENT_INPUT)[0]
        
        # Path 2: If only image is connected, use the image directly
        elif IMAGE is not None:
            output_image = IMAGE
        
        # If no image or latent input, create a default black image
        if output_image is None:
            output_image = torch.zeros(1, 64, 64, 3)
        
        # Upscale the image
        if output_image is not None:
            # Load upscale model
            upscale_model = None
            if UPSCALE_MODEL != "Default":
                upscale_model = UpscaleModelLoader().load_model(UPSCALE_MODEL)[0]
            
            # Upscale the image if a model is available
            if upscale_model is not None:
                output_image = ImageUpscaleWithModel().upscale(upscale_model, output_image)[0]
            
            # Resize the image based on longest side
            assert isinstance(output_image, torch.Tensor)
            assert isinstance(OUTPUT_LONGEST_SIDE, int)
            assert isinstance(INTERPOLATION_MODE, str)
            
            INTERPOLATION_MODE = INTERPOLATION_MODE.upper().replace(" ", "_")
            INTERPOLATION_MODE = getattr(InterpolationMode, INTERPOLATION_MODE)
            _, h, w, _ = output_image.shape
            if h >= w:
                new_h = OUTPUT_LONGEST_SIDE
                new_w = round(w * new_h / h)
            else:  # h < w
                new_w = OUTPUT_LONGEST_SIDE
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