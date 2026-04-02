import torch
import numpy as np

class StarImageSwitch2:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img_out",)
    FUNCTION = "process_images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "Image 1": ("IMAGE",),
                "Image 2": ("IMAGE",),
                "Image 3": ("IMAGE",),
                "Image 4": ("IMAGE",),
                "Image 5": ("IMAGE",),
            }
        }

    def process_images(self, **kwargs):
        for i in range(1, 6):
            img = kwargs.get(f"Image {i}")
            if img is not None:
                return (img,)
        h, w = 512, 512
        default_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
        for i in range(0, h, 32):
            for j in range(0, w, 32):
                if (i + j) % 64 == 0:
                    default_img[i:i+16, j:j+16] = 0.7
        img_tensor = torch.from_numpy(default_img)
        img_tensor = img_tensor.unsqueeze(0)
        return (img_tensor,)

NODE_CLASS_MAPPINGS = {
    "StarImageSwitch2": StarImageSwitch2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageSwitch2": "⭐ Star Image Input 2 (Optimized)"
}
