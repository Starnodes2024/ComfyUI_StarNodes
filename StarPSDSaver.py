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

class StarPSDSaver:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"    # Title color
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "multilayer"}),
                "output_dir": ("STRING", {"default": "PSD_Layers"}),
            },
            "optional": {
                "layer1": ("IMAGE",),
                "mask1": ("MASK",),
            }
        }

    def tensor_to_pil(self, image_tensor):
        """Convert a PyTorch tensor to a PIL Image."""
        if image_tensor is None:
            return None
            
        # If tensor has batch dimension, take the first image
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
            
        # Convert to numpy and scale to 0-255 range
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert from RGB to PIL Image
        return Image.fromarray(image_np)
    
    def tensor_to_mask(self, mask_tensor):
        """Convert a PyTorch tensor to a PIL Image mask."""
        if mask_tensor is None:
            return None
            
        # If tensor has batch dimension, take the first image
        if len(mask_tensor.shape) == 4:
            mask_tensor = mask_tensor[0]
        
        # If the mask has 3 channels, convert to grayscale
        if mask_tensor.shape[0] == 3:
            # Use the average of RGB channels
            mask_tensor = torch.mean(mask_tensor, dim=0, keepdim=True)
        
        # Ensure mask is single channel
        if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor[0]
            
        # Convert to numpy and scale to 0-255 range
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image (L mode for grayscale)
        return Image.fromarray(mask_np, mode='L')

    def save_psd(self, filename_prefix, output_dir, **kwargs):
        """Save multiple image layers as a PSD file with masks."""
        
        # Get ComfyUI's standard output directory as base
        base_output_dir = folder_paths.get_output_directory()
        
        # Create full output path (user's subdirectory under ComfyUI output folder)
        full_output_dir = os.path.join(base_output_dir, output_dir)
        
        print(f"[StarPSDSaver] Base output dir: {base_output_dir}")
        print(f"[StarPSDSaver] User subdir: {output_dir}")
        print(f"[StarPSDSaver] Full output path: {full_output_dir}")
        
        # Ensure output directory exists
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
        
        # Collect all connected layers and masks
        layers = []
        masks = []
        
        # Find all layer inputs in kwargs
        layer_inputs = {k: v for k, v in kwargs.items() if k.startswith("layer") and v is not None}
        
        # Sort layer inputs by number
        sorted_layer_keys = sorted(layer_inputs.keys(), 
                                  key=lambda x: int(x.replace("layer", "")))
        
        # Find the maximum width and height among all layers
        max_width = 0
        max_height = 0
        
        for layer_key in sorted_layer_keys:
            img_tensor = kwargs.get(layer_key)
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                if pil_img:
                    # Update max dimensions
                    width, height = pil_img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    
                    # Get corresponding mask if it exists
                    layer_num = layer_key.replace("layer", "")
                    mask_key = f"mask{layer_num}"
                    mask_tensor = kwargs.get(mask_key)
                    
                    mask_pil = None
                    if mask_tensor is not None:
                        mask_pil = self.tensor_to_mask(mask_tensor)
                        
                        # Ensure mask has same dimensions as the image
                        if mask_pil and mask_pil.size != pil_img.size:
                            mask_pil = mask_pil.resize(pil_img.size, Image.LANCZOS)
                    
                    layers.append(pil_img)
                    masks.append(mask_pil)
        
        if not layers:
            print("No layers provided to save as PSD.")
            return ()
        
        # Create a new PSD file with the maximum dimensions
        psd = PSDImage.new(mode='RGB', size=(max_width, max_height))
        
        # Add layers from bottom to top (reverse order for PSD)
        for i, (pil_img, mask_pil) in enumerate(zip(layers, masks)):
            # Ensure the image is in RGB mode
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
                
            # If the image is smaller than the PSD, center it
            if pil_img.size != (max_width, max_height):
                # Create a new image with the max dimensions
                new_img = Image.new('RGB', (max_width, max_height), (0, 0, 0))
                # Calculate position to paste (centered)
                x_offset = (max_width - pil_img.width) // 2
                y_offset = (max_height - pil_img.height) // 2
                # Paste the original image
                new_img.paste(pil_img, (x_offset, y_offset))
                pil_img = new_img
                
                # If there's a mask, center it too
                if mask_pil:
                    new_mask = Image.new('L', (max_width, max_height), 0)
                    new_mask.paste(mask_pil, (x_offset, y_offset))
                    mask_pil = new_mask
            
            # For layers with masks, we need to create a layer with transparency
            if mask_pil:
                # Create a regular layer first
                layer = PixelLayer.frompil(pil_img, psd, f"Layer {i+1}")
                
                # Add the layer to the PSD
                psd.append(layer)
                
                # Create a mask for the layer
                # The mask needs to be in grayscale mode
                if mask_pil.mode != 'L':
                    mask_pil = mask_pil.convert('L')
                
                # Create the mask data structure
                mask_data = MaskData(
                    top=0, left=0, 
                    bottom=mask_pil.height, right=mask_pil.width,
                    background_color=0,
                    flags=MaskFlags(parameters_applied=True)
                )
                
                # Set the mask data directly on the layer record
                layer._record.mask_data = mask_data
                
                # Add the mask channel to the layer's channels
                # We need to use the proper channel data structure
                channel_data = ChannelData(compression=Compression.RAW)
                # Use the set_data method with the correct parameters
                channel_data.set_data(
                    mask_pil.tobytes(),  # raw data bytes
                    mask_pil.width,      # width
                    mask_pil.height,     # height
                    8,                   # bit depth
                    1                    # version (default)
                )
                
                # Add the channel info to the layer
                layer._record.channel_info.append(
                    ChannelInfo(id=ChannelID.USER_LAYER_MASK, length=0)  # Length will be updated when saved
                )
                
                # Add the channel data to the layer's channels
                layer._channels.append(channel_data)
            else:
                # Create a regular RGB layer
                layer = PixelLayer.frompil(pil_img, psd, f"Layer {i+1}")
                psd.append(layer)
        
        # Save the PSD file
        psd.save(save_path)
        print(f"PSD file saved to {save_path}")
        
        return ()

NODE_CLASS_MAPPINGS = {
    "StarPSDSaver": StarPSDSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPSDSaver": "⭐ Star PSD Saver (Dynamic)"
}
