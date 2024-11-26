import random

class StarNode:
    CATEGORY = 'StarNodes'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img_out",)
    FUNCTION = "process_images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "SDXL = 1": ("IMAGE",),
                "AURAFLOW = 2": ("IMAGE",),
                "SD3 = 3": ("IMAGE",),
                "PIXART SIGMA = 4": ("IMAGE",),
                "KOLORS = 5": ("IMAGE",),
                "STABLE CASCADE = 6": ("IMAGE",),
                "FREE = 7": ("IMAGE",),
                "FREE = 8": ("IMAGE",),
            }
        }

    def process_images(self, select, **kwargs):
        images = [
            kwargs.get("SDXL = 1"),
            kwargs.get("AURAFLOW = 2"),
            kwargs.get("SD3 = 3"),
            kwargs.get("PIXART SIGMA = 4"),
            kwargs.get("KOLORS = 5"),
            kwargs.get("STABLE CASCADE = 6"),
            kwargs.get("FREE = 7"),
            kwargs.get("FREE = 8")
        ]
        
        if 1 <= select <= 8:
            img_out = images[select - 1]
        else:
            img_out = images[0]  # Default to first image if selection is out of range
        
        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "StarNode": StarNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarNode": "⭐ StarNode Input Image Chooser"
}