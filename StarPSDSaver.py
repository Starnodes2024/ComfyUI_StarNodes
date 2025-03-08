import os
import torch
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import ColorMode

class StarPSDSaver:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"    # Title color
    CATEGORY = '⭐StarNodes'
    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "multilayer"}),
                "output_dir": ("STRING", {"default": "ComfyUI/output/PSD_Layers"}),
            },
            "optional": {
                "layer1": ("IMAGE",),
                "layer2": ("IMAGE",),
                "layer3": ("IMAGE",),
                "layer4": ("IMAGE",),
                "layer5": ("IMAGE",),
                "layer6": ("IMAGE",),
                "layer7": ("IMAGE",),
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

    def save_psd(self, filename_prefix, output_dir, 
                 layer1=None, layer2=None, layer3=None, layer4=None, layer5=None, layer6=None, layer7=None):
        """Save multiple image layers as a PSD file."""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        counter = 1
        while True:
            if counter == 1:
                filename = f"{filename_prefix}.psd"
            else:
                filename = f"{filename_prefix}_{counter}.psd"
                
            save_path = os.path.join(output_dir, filename)
            if not os.path.exists(save_path):
                break
            counter += 1
        
        # Collect all connected layers
        layers = []
        layer_images = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
        
        # Find the maximum width and height among all layers
        max_width = 0
        max_height = 0
        
        for i, img_tensor in enumerate(layer_images):
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                if pil_img:
                    # Update max dimensions
                    width, height = pil_img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    
                    layers.append(pil_img)
        
        if not layers:
            print("No layers provided to save as PSD.")
            return ()
        
        # Create a new PSD file with the maximum dimensions
        psd = PSDImage.new(mode='RGB', size=(max_width, max_height))
        
        # Add layers from bottom to top (reverse order for PSD)
        for i, pil_img in enumerate(reversed(layers)):
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
            
            # Create a new layer in the PSD
            layer = PixelLayer.frompil(pil_img, psd)
            psd.append(layer)
        
        # Save the PSD file
        psd.save(save_path)
        print(f"PSD file saved to {save_path}")
        
        return ()

NODE_CLASS_MAPPINGS = {
    "StarPSDSaver": StarPSDSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPSDSaver": "⭐ Star 7 Layers 2 PSD"
}
