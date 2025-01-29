import torch
import numpy as np

class StarImageSwitch:
    CATEGORY = '⭐StarNodes'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img_out",)
    FUNCTION = "process_images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "Image 1": ("IMAGE",),
                "Image 2": ("IMAGE",),
                "Image 3": ("IMAGE",),
                "Image 4": ("IMAGE",),
                "Image 5": ("IMAGE",),
                "Image 6": ("IMAGE",),
                "Image 7": ("IMAGE",),
            }
        }

    def process_images(self, **kwargs):
        # Try to get the first connected image
        for i in range(1, 8):
            img = kwargs.get(f"Image {i}")
            if img is not None:
                return (img,)
        
        # If no image is connected, create a default gray image with text pattern
        h, w = 512, 512
        # Create a gray image with a gradient
        default_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
        # Add some visual pattern
        for i in range(0, h, 32):
            for j in range(0, w, 32):
                if (i + j) % 64 == 0:
                    default_img[i:i+16, j:j+16] = 0.7
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(default_img)
        img_tensor = img_tensor.unsqueeze(0)
        return (img_tensor,)

NODE_CLASS_MAPPINGS = {
    "StarImageSwitch": StarImageSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageSwitch": "⭐ Star Seven Inputs (img)"
}