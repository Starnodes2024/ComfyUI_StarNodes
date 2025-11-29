import os
import torch
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import ColorMode, ChannelID, Compression, BlendMode
from psd_tools.api.mask import Mask
from psd_tools.psd.layer_and_mask import MaskData, MaskFlags, ChannelInfo, ChannelData
from psd_tools.compression import compress
import folder_paths


# Photoshop blend modes mapping
BLEND_MODES = [
    "normal",
    "dissolve",
    "darken",
    "multiply",
    "color_burn",
    "linear_burn",
    "darker_color",
    "lighten",
    "screen",
    "color_dodge",
    "linear_dodge",
    "lighter_color",
    "overlay",
    "soft_light",
    "hard_light",
    "vivid_light",
    "linear_light",
    "pin_light",
    "hard_mix",
    "difference",
    "exclusion",
    "subtract",
    "divide",
    "hue",
    "saturation",
    "color",
    "luminosity",
]

# Map blend mode names to psd_tools BlendMode constants
BLEND_MODE_MAP = {
    "normal": BlendMode.NORMAL,
    "dissolve": BlendMode.DISSOLVE,
    "darken": BlendMode.DARKEN,
    "multiply": BlendMode.MULTIPLY,
    "color_burn": BlendMode.COLOR_BURN,
    "linear_burn": BlendMode.LINEAR_BURN,
    "darker_color": BlendMode.DARKER_COLOR,
    "lighten": BlendMode.LIGHTEN,
    "screen": BlendMode.SCREEN,
    "color_dodge": BlendMode.COLOR_DODGE,
    "linear_dodge": BlendMode.LINEAR_DODGE,
    "lighter_color": BlendMode.LIGHTER_COLOR,
    "overlay": BlendMode.OVERLAY,
    "soft_light": BlendMode.SOFT_LIGHT,
    "hard_light": BlendMode.HARD_LIGHT,
    "vivid_light": BlendMode.VIVID_LIGHT,
    "linear_light": BlendMode.LINEAR_LIGHT,
    "pin_light": BlendMode.PIN_LIGHT,
    "hard_mix": BlendMode.HARD_MIX,
    "difference": BlendMode.DIFFERENCE,
    "exclusion": BlendMode.EXCLUSION,
    "subtract": BlendMode.SUBTRACT,
    "divide": BlendMode.DIVIDE,
    "hue": BlendMode.HUE,
    "saturation": BlendMode.SATURATION,
    "color": BlendMode.COLOR,
    "luminosity": BlendMode.LUMINOSITY,
}


class FlexibleLayerInputs(dict):
    """A special class to make flexible layer inputs for the advanced PSD saver."""
    def __init__(self, base_inputs=None):
        super().__init__()
        if base_inputs:
            self.update(base_inputs)

    def __getitem__(self, key):
        # Check if it's in the base dict first
        if key in self.keys():
            return super().__getitem__(key)
        # Dynamic inputs
        if key.startswith("layer"):
            return ("IMAGE",)
        elif key.startswith("mask"):
            return ("MASK",)
        elif key.startswith("blend_mode"):
            return (BLEND_MODES, {"default": "normal"})
        elif key.startswith("opacity"):
            return ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0})
        return ("STRING",)

    def __contains__(self, key):
        return True


class StarPSDSaverAdvLayers:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flattened_image",)
    FUNCTION = "save_psd_adv"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "multilayer_adv"}),
                "output_dir": ("STRING", {"default": "PSD_Layers"}),
                "save_psd": ("BOOLEAN", {"default": True}),
            },
            "optional": FlexibleLayerInputs({
                "layer1": ("IMAGE",),
                "mask1": ("MASK",),
            }),
        }

    def tensor_to_pil(self, image_tensor):
        """Convert a PyTorch tensor to a PIL Image."""
        if image_tensor is None:
            return None
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_tensor(self, pil_image):
        """Convert a PIL Image to a PyTorch tensor."""
        if pil_image is None:
            return None
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def tensor_to_mask(self, mask_tensor):
        """Convert a PyTorch tensor to a PIL Image mask."""
        if mask_tensor is None:
            return None
        if len(mask_tensor.shape) == 4:
            mask_tensor = mask_tensor[0]
        if mask_tensor.shape[0] == 3:
            mask_tensor = torch.mean(mask_tensor, dim=0, keepdim=True)
        if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor[0]
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, mode='L')

    def apply_blend_mode(self, base, layer, blend_mode, opacity):
        """Apply blend mode and opacity to composite layer onto base."""
        # Convert to float for blending calculations
        base_arr = np.array(base).astype(np.float32) / 255.0
        layer_arr = np.array(layer).astype(np.float32) / 255.0
        
        opacity_factor = opacity / 100.0
        
        # Apply blend mode
        if blend_mode == "normal":
            result = layer_arr
        elif blend_mode == "multiply":
            result = base_arr * layer_arr
        elif blend_mode == "screen":
            result = 1.0 - (1.0 - base_arr) * (1.0 - layer_arr)
        elif blend_mode == "overlay":
            mask = base_arr < 0.5
            result = np.where(mask, 2 * base_arr * layer_arr, 1 - 2 * (1 - base_arr) * (1 - layer_arr))
        elif blend_mode == "darken":
            result = np.minimum(base_arr, layer_arr)
        elif blend_mode == "lighten":
            result = np.maximum(base_arr, layer_arr)
        elif blend_mode == "color_dodge":
            result = np.where(layer_arr >= 1.0, 1.0, np.minimum(1.0, base_arr / (1.0 - layer_arr + 1e-6)))
        elif blend_mode == "color_burn":
            result = np.where(layer_arr <= 0.0, 0.0, np.maximum(0.0, 1.0 - (1.0 - base_arr) / (layer_arr + 1e-6)))
        elif blend_mode == "hard_light":
            mask = layer_arr < 0.5
            result = np.where(mask, 2 * base_arr * layer_arr, 1 - 2 * (1 - base_arr) * (1 - layer_arr))
        elif blend_mode == "soft_light":
            result = np.where(layer_arr < 0.5,
                              base_arr - (1 - 2 * layer_arr) * base_arr * (1 - base_arr),
                              base_arr + (2 * layer_arr - 1) * (np.sqrt(base_arr) - base_arr))
        elif blend_mode == "difference":
            result = np.abs(base_arr - layer_arr)
        elif blend_mode == "exclusion":
            result = base_arr + layer_arr - 2 * base_arr * layer_arr
        elif blend_mode == "add" or blend_mode == "linear_dodge":
            result = np.minimum(1.0, base_arr + layer_arr)
        elif blend_mode == "subtract":
            result = np.maximum(0.0, base_arr - layer_arr)
        elif blend_mode == "divide":
            result = np.where(layer_arr <= 0.0, 1.0, np.minimum(1.0, base_arr / (layer_arr + 1e-6)))
        elif blend_mode == "linear_burn":
            result = np.maximum(0.0, base_arr + layer_arr - 1.0)
        elif blend_mode == "vivid_light":
            mask = layer_arr < 0.5
            result = np.where(mask,
                              np.where(layer_arr <= 0.0, 0.0, np.maximum(0.0, 1.0 - (1.0 - base_arr) / (2 * layer_arr + 1e-6))),
                              np.where(layer_arr >= 1.0, 1.0, np.minimum(1.0, base_arr / (2 * (1.0 - layer_arr) + 1e-6))))
        elif blend_mode == "linear_light":
            result = np.clip(base_arr + 2 * layer_arr - 1.0, 0.0, 1.0)
        elif blend_mode == "pin_light":
            result = np.where(layer_arr < 0.5,
                              np.minimum(base_arr, 2 * layer_arr),
                              np.maximum(base_arr, 2 * layer_arr - 1))
        elif blend_mode == "hard_mix":
            result = np.where(base_arr + layer_arr >= 1.0, 1.0, 0.0)
        else:
            # Default to normal for unsupported modes
            result = layer_arr
        
        # Apply opacity: blend result with base
        blended = base_arr * (1 - opacity_factor) + result * opacity_factor
        
        # Clip and convert back to uint8
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def apply_blend_mode_with_mask(self, base, layer, mask, blend_mode, opacity):
        """Apply blend mode with mask support."""
        # First apply the blend mode
        blended = self.apply_blend_mode(base, layer, blend_mode, opacity)
        
        if mask is None:
            return blended
        
        # Apply mask
        base_arr = np.array(base).astype(np.float32)
        blended_arr = np.array(blended).astype(np.float32)
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        
        # Expand mask to 3 channels if needed
        if len(mask_arr.shape) == 2:
            mask_arr = np.stack([mask_arr] * 3, axis=-1)
        
        # Blend using mask
        result = base_arr * (1 - mask_arr) + blended_arr * mask_arr
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def save_psd_adv(self, filename_prefix, output_dir, save_psd=True, **kwargs):
        """Save multiple image layers as a PSD file with blend modes and opacity.

        If save_psd is False, no PSD file will be written to disk and only the
        flattened image will be returned as an IMAGE tensor.
        """
        
        save_path = None
        if save_psd:
            # Get ComfyUI's standard output directory as base
            base_output_dir = folder_paths.get_output_directory()
            full_output_dir = os.path.join(base_output_dir, output_dir)
            
            print(f"[StarPSDSaverAdvLayers] Full output path: {full_output_dir}")
            os.makedirs(full_output_dir, exist_ok=True)
            
            # Generate a unique filename
            counter = 1
            while True:
                if counter == 1:
                    filename = f"{filename_prefix}.psd"
                else:
                    filename = f"{filename_prefix}_{counter}.psd"
                save_path = os.path.join(full_output_dir, filename)
                if not os.path.exists(save_path):
                    break
                counter += 1
        
        # Collect all layer data
        layer_data = []
        
        # Find all layer inputs in kwargs
        layer_keys = sorted(
            [k for k in kwargs.keys() if k.startswith("layer") and kwargs[k] is not None],
            key=lambda x: int(x.replace("layer", ""))
        )
        
        max_width = 0
        max_height = 0
        
        for layer_key in layer_keys:
            layer_num = layer_key.replace("layer", "")
            img_tensor = kwargs.get(layer_key)
            
            if img_tensor is None:
                continue
            
            pil_img = self.tensor_to_pil(img_tensor)
            if pil_img is None:
                continue
            
            width, height = pil_img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            
            # Get corresponding mask, blend mode, and opacity
            mask_tensor = kwargs.get(f"mask{layer_num}")
            blend_mode = kwargs.get(f"blend_mode{layer_num}", "normal")
            opacity = kwargs.get(f"opacity{layer_num}", 100.0)
            
            mask_pil = None
            if mask_tensor is not None:
                mask_pil = self.tensor_to_mask(mask_tensor)
                if mask_pil and mask_pil.size != pil_img.size:
                    mask_pil = mask_pil.resize(pil_img.size, Image.LANCZOS)
            
            layer_data.append({
                "image": pil_img,
                "mask": mask_pil,
                "blend_mode": blend_mode if blend_mode else "normal",
                "opacity": float(opacity) if opacity is not None else 100.0,
                "name": f"Layer {layer_num}"
            })
        
        if not layer_data:
            print("[StarPSDSaverAdvLayers] No layers provided.")
            # Return empty black image
            empty = torch.zeros(1, 64, 64, 3)
            return (empty,)
        
        # Create PSD container only if we will save a PSD file
        psd = None
        if save_psd:
            psd = PSDImage.new(mode='RGB', size=(max_width, max_height))
        
        # Create flattened composite
        flattened = Image.new('RGB', (max_width, max_height), (0, 0, 0))
        
        for i, data in enumerate(layer_data):
            pil_img = data["image"]
            mask_pil = data["mask"]
            blend_mode = data["blend_mode"]
            opacity = data["opacity"]
            layer_name = data["name"]
            
            # Ensure RGB mode
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Center smaller images
            if pil_img.size != (max_width, max_height):
                new_img = Image.new('RGB', (max_width, max_height), (0, 0, 0))
                x_offset = (max_width - pil_img.width) // 2
                y_offset = (max_height - pil_img.height) // 2
                new_img.paste(pil_img, (x_offset, y_offset))
                pil_img = new_img
                
                if mask_pil:
                    new_mask = Image.new('L', (max_width, max_height), 0)
                    new_mask.paste(mask_pil, (x_offset, y_offset))
                    mask_pil = new_mask
            
            # Update flattened composite (first layer is base, others blend on top)
            if i == 0:
                if mask_pil:
                    # Apply mask to first layer
                    flattened_arr = np.array(flattened).astype(np.float32)
                    layer_arr = np.array(pil_img).astype(np.float32)
                    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
                    if len(mask_arr.shape) == 2:
                        mask_arr = np.stack([mask_arr] * 3, axis=-1)
                    result = flattened_arr * (1 - mask_arr) + layer_arr * mask_arr
                    flattened = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
                else:
                    flattened = pil_img.copy()
            else:
                flattened = self.apply_blend_mode_with_mask(flattened, pil_img, mask_pil, blend_mode, opacity)
            
            # Create PSD layer (only if saving PSD)
            if save_psd and psd is not None:
                layer = PixelLayer.frompil(pil_img, psd, layer_name)
                
                # Set blend mode
                psd_blend_mode = BLEND_MODE_MAP.get(blend_mode, BlendMode.NORMAL)
                layer._record.blend_mode = psd_blend_mode
                
                # Set opacity (0-255 in PSD)
                layer._record.opacity = int(opacity * 255 / 100)
                
                psd.append(layer)
                
                # Add mask if present
                if mask_pil:
                    if mask_pil.mode != 'L':
                        mask_pil = mask_pil.convert('L')
                    
                    mask_data = MaskData(
                        top=0, left=0,
                        bottom=mask_pil.height, right=mask_pil.width,
                        background_color=0,
                        flags=MaskFlags(parameters_applied=True)
                    )
                    layer._record.mask_data = mask_data
                    
                    channel_data = ChannelData(compression=Compression.RAW)
                    channel_data.set_data(
                        mask_pil.tobytes(),
                        mask_pil.width,
                        mask_pil.height,
                        8,
                        1
                    )
                    layer._record.channel_info.append(
                        ChannelInfo(id=ChannelID.USER_LAYER_MASK, length=0)
                    )
                    layer._channels.append(channel_data)
        
        # Save PSD only if requested
        if save_psd and psd is not None and save_path is not None:
            psd.save(save_path)
            print(f"[StarPSDSaverAdvLayers] PSD saved to {save_path}")
        
        # Convert flattened to tensor
        flattened_tensor = self.pil_to_tensor(flattened)
        
        return (flattened_tensor,)


NODE_CLASS_MAPPINGS = {
    "StarPSDSaverAdvLayers": StarPSDSaverAdvLayers
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPSDSaverAdvLayers": "⭐ Star PSD Saver Adv. Layers"
}
