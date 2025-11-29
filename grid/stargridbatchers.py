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
        optional_inputs = {"image_batch": ("IMAGE",), "image 1": ("IMAGE",)}
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
        for i in range(1, 101):
            img_key = f"image {i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                images.append(kwargs[img_key])
        
        # If no images provided, return empty batch
        if not images:
            return (torch.zeros((0, 3, 64, 64), dtype=torch.float32),)
        
        # Find the largest side among all images
        max_side = 0
        processed_images = []
        for img in images:
            # Convert to torch if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            # If shape is [B, H, W, C], permute to [B, C, H, W]
            if img.shape[-1] == 3 and img.dim() == 4:
                img = img.permute(0, 3, 1, 2)
            elif img.shape[1] == 3 and img.dim() == 4:
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            b, c, h, w = img.shape
            max_side = max(max_side, h, w)
            processed_images.append(img)
        # Now make all images square, centered, using the longest side
        squared_images = []
        for img in processed_images:
            b, c, h, w = img.shape
            # Remove batch dimension for resizing, will add back later
            img = img[0]
            scale = max_side / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            img_resized = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
            # Pad to make square and centered
            pad_top = (max_side - new_h) // 2
            pad_bottom = max_side - new_h - pad_top
            pad_left = (max_side - new_w) // 2
            pad_right = max_side - new_w - pad_left
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            img_padded = torch.nn.functional.pad(img_resized, padding, mode='constant', value=0)
            squared_images.append(img_padded.unsqueeze(0))
        result = torch.cat(squared_images, dim=0)
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
        optional_inputs = {"caption 1": ("STRING", {"forceInput": True})}
        return {
            "required": {},
            "optional": optional_inputs
        }
    
    def batch_captions(self, **kwargs):
        # Collect all captions
        captions = []
        
        # Check for individual inputs
        for i in range(1, 101):
            caption_key = f"caption {i}"
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
