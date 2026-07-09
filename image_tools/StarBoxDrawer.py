import torch
import numpy as np
from PIL import Image, ImageDraw


class StarBoxDrawer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "width": ("INT", {"default": 100, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 100, "min": 1, "max": 8192}),
                "color": (["white", "red", "blue", "green", "black"], {"default": "white"}),
                "filled": ("BOOLEAN", {"default": True}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_box"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def draw_box(self, image, x, y, width, height, color, filled, line_width):
        color_map = {
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "black": (0, 0, 0),
        }
        rgb_color = color_map[color]

        batch_size = image.shape[0]
        result = []

        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            draw = ImageDraw.Draw(img_pil)
            if filled:
                draw.rectangle([x, y, x + width, y + height], fill=rgb_color)
            else:
                draw.rectangle([x, y, x + width, y + height], outline=rgb_color, width=line_width)
            
            img_result = np.array(img_pil).astype(np.float32) / 255.0
            result.append(img_result)

        result_tensor = torch.from_numpy(np.stack(result))
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "StarBoxDrawer": StarBoxDrawer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarBoxDrawer": "⭐ Star Box Drawer",
}
