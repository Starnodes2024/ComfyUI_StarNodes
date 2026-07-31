import torch
import numpy as np

class StarShowLastFrame:
    """
    Node to show only the last frame of an image batch.
    """
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    def process(self, images):
        # Extract the last image from the batch
        if len(images.shape) == 4 and images.shape[0] > 0:
            # Get the last image from the batch
            last_image = images[-1:].clone()
            return (last_image,)
        else:
            # If there's only one image or an empty batch, return it as is
            return (images,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

# For dynamic import
NODE_CLASS_MAPPINGS = {"Star_Show_Last_Frame": StarShowLastFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"Star_Show_Last_Frame": "⭐ Star Show Last Frame"}
