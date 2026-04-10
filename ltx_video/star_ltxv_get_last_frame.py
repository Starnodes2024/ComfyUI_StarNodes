import torch
import numpy as np

class StarLTXVGetLastFrame:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "get_last_frame"
    CATEGORY = "⭐StarNodes/LTX Video"
    
    def get_last_frame(self, images):
        if isinstance(images, dict):
            raise ValueError(f"Unexpected dict input. Expected IMAGE tensor, got dict with keys: {images.keys()}")
        
        if isinstance(images, torch.Tensor):
            n = images.shape[0]
        elif isinstance(images, np.ndarray):
            n = images.shape[0]
        elif isinstance(images, list):
            n = len(images)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}. Expected torch.Tensor, numpy.ndarray, or list.")
        
        if n == 0:
            raise ValueError("Image batch is empty.")
        
        idx = n - 1
        
        if isinstance(images, torch.Tensor):
            last_frame = images[idx:idx+1]
        elif isinstance(images, np.ndarray):
            last_frame = images[idx:idx+1]
        else:
            last_frame = [images[idx]]
        
        return (last_frame,)


NODE_CLASS_MAPPINGS = {
    "StarLTXVGetLastFrame": StarLTXVGetLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLTXVGetLastFrame": "⭐ Star LTXV Get Last Frame",
}
