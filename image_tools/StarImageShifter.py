import torch
import numpy as np
from PIL import Image


class StarImageShifter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x_shift": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "y_shift": ("INT", {"default": 0, "min": -8192, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shift_image"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def shift_image(self, image, x_shift, y_shift):
        batch_size = image.shape[0]
        result = []

        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            width, height = img_pil.size
            
            shifted = Image.new('RGB', (width, height))
            
            x_shift_wrapped = x_shift % width
            y_shift_wrapped = y_shift % height
            
            shifted.paste(img_pil.crop((width - x_shift_wrapped, 0, width, height)), (0, 0))
            shifted.paste(img_pil.crop((0, 0, width - x_shift_wrapped, height)), (x_shift_wrapped, 0))
            
            temp = shifted.copy()
            shifted.paste(temp.crop((0, height - y_shift_wrapped, width, height)), (0, 0))
            shifted.paste(temp.crop((0, 0, width, height - y_shift_wrapped)), (0, y_shift_wrapped))
            
            img_result = np.array(shifted).astype(np.float32) / 255.0
            result.append(img_result)

        result_tensor = torch.from_numpy(np.stack(result))
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "StarImageShifter": StarImageShifter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageShifter": "⭐ Star Image Shifter",
}
