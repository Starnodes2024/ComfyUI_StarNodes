import torch
import numpy as np

class StarGridImageBatcher:
    """
    Batches multiple images together for use with the Star Grid Composer.
    Accepts individual images or an image batch and combines them into a single batch.
    """
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Grid Image Batch",)
    FUNCTION = "batch_images"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            "image_batch": ("IMAGE",),
        }
        
        # Add image_1 through image_16
        for i in range(1, 17):
            optional_inputs[f"image_{i}"] = ("IMAGE",)
            
        return {
            "required": {},
            "optional": optional_inputs
        }
    
    def batch_images(self, **kwargs):
        # Collect all valid images
        images = []
        
        # First check for batch input
        if "image_batch" in kwargs and kwargs["image_batch"] is not None:
            batch = kwargs["image_batch"]
            if len(batch.shape) == 4:  # [B, H, W, C] format
                for i in range(batch.shape[0]):
                    images.append(batch[i:i+1])
        
        # Then check for individual inputs
        for i in range(1, 17):
            img_key = f"image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                images.append(kwargs[img_key])
        
        # If no images provided, return empty batch
        if not images:
            return (torch.zeros((0, 3, 64, 64), dtype=torch.float32),)
        
        # Combine all images into a single batch
        result = torch.cat(images, dim=0)
        return (result,)


class StarGridCaptionsBatcher:
    """
    Batches multiple captions together for use with the Star Grid Composer.
    Accepts individual caption strings and combines them into a single string.
    """
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Grid Captions Batch",)
    FUNCTION = "batch_captions"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        
        # Add caption_1 through caption_16 as input connectors only
        for i in range(1, 17):
            optional_inputs[f"caption_{i}"] = ("STRING", {"forceInput": True})
            
        return {
            "required": {},
            "optional": optional_inputs
        }
    
    def batch_captions(self, **kwargs):
        # Collect all captions
        captions = []
        
        # Check for individual inputs
        for i in range(1, 17):
            caption_key = f"caption_{i}"
            if caption_key in kwargs and kwargs[caption_key]:
                captions.append(kwargs[caption_key])
            else:
                # Add empty caption to maintain position
                captions.append("")
        
        # Remove trailing empty captions
        while captions and captions[-1] == "":
            captions.pop()
        
        # Join with newlines
        result = "\n".join(captions)
        return (result,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "StarGridImageBatcher": StarGridImageBatcher,
    "StarGridCaptionsBatcher": StarGridCaptionsBatcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarGridImageBatcher": "⭐ Star Grid Image Batcher",
    "StarGridCaptionsBatcher": "⭐ Star Grid Captions Batcher"
}
