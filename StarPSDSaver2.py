import os
import torch
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import ColorMode, ChannelID, Compression
from psd_tools.api.mask import Mask
from psd_tools.psd.layer_and_mask import MaskData, MaskFlags, ChannelInfo, ChannelData
from psd_tools.compression import compress
import folder_paths

class StarPSDSaver2:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"    # Title color
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # Accept up to 10 layers and masks as optional inputs
        optional_inputs = {}
        for i in range(1, 11):
            optional_inputs[f"layer{i}"] = ("IMAGE",)
            optional_inputs[f"mask{i}"] = ("MASK",)
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "multilayer2"}),
                "output_dir": ("STRING", {"default": "PSD_Layers"}),
            },
            "optional": optional_inputs
        }

    def tensor_to_pil(self, image_tensor):
        if image_tensor is None:
            return None
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        return Image.fromarray(image_np)

    def tensor_to_mask(self, mask_tensor):
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

    def save_psd(self, filename_prefix, output_dir, **kwargs):
        # Get ComfyUI's standard output directory as base
        base_output_dir = folder_paths.get_output_directory()
        
        # Create full output path (user's subdirectory under ComfyUI output folder)
        full_output_dir = os.path.join(base_output_dir, output_dir)
        
        print(f"[StarPSDSaver2] Base output dir: {base_output_dir}")
        print(f"[StarPSDSaver2] User subdir: {output_dir}")
        print(f"[StarPSDSaver2] Full output path: {full_output_dir}")
        
        # Ensure output directory exists
        os.makedirs(full_output_dir, exist_ok=True)
        counter = 1
        while True:
            filename = f"{filename_prefix}.psd" if counter == 1 else f"{filename_prefix}_{counter}.psd"
            save_path = os.path.join(full_output_dir, filename)
            if not os.path.exists(save_path):
                break
            counter += 1
        layers = []
        masks = []
        max_width = 0
        max_height = 0
        # Collect all layer/mask pairs in order
        for i in range(1, 11):
            img_tensor = kwargs.get(f"layer{i}")
            mask_tensor = kwargs.get(f"mask{i}")
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                if pil_img:
                    width, height = pil_img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    mask_pil = None
                    if mask_tensor is not None:
                        mask_pil = self.tensor_to_mask(mask_tensor)
                        if mask_pil and mask_pil.size != pil_img.size:
                            mask_pil = mask_pil.resize(pil_img.size, Image.LANCZOS)
                    layers.append(pil_img)
                    masks.append(mask_pil)
        if not layers:
            print("No layers provided to save as PSD.")
            return ()
        psd = PSDImage.new(mode='RGB', size=(max_width, max_height))
        for i, (pil_img, mask_pil) in enumerate(zip(layers, masks)):
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
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
            if mask_pil:
                layer = PixelLayer.frompil(pil_img, psd, f"Layer {i+1}")
                psd.append(layer)
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
            else:
                layer = PixelLayer.frompil(pil_img, psd, f"Layer {i+1}")
                psd.append(layer)
        psd.save(save_path)
        print(f"PSD file saved to {save_path}")
        return ()

NODE_CLASS_MAPPINGS = {
    "StarPSDSaver2": StarPSDSaver2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPSDSaver2": "⭐ Star PSD Saver 2 (Optimized)"
}
