import random

class StarImageSwitch:
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
                "Image 1 = 1": ("IMAGE",),
                "Image 2 = 2": ("IMAGE",),
                "Image 3 = 3": ("IMAGE",),
                "Image 4  = 4": ("IMAGE",),
                "Image 5  = 5": ("IMAGE",),
                "Image 6  = 6": ("IMAGE",),
                "Image 7  = 7": ("IMAGE",),
                "Image 8  = 8": ("IMAGE",),
            }
        }

    def process_images(self, select, **kwargs):
        images = [
            kwargs.get("Image 1 = 1"),
            kwargs.get("Image 2 = 2"),
            kwargs.get("Image 3 = 3"),
            kwargs.get("Image 4 = 4"),
            kwargs.get("Image 4 = 5"),
            kwargs.get("Image 5 = 6"),
            kwargs.get("Image 6 = 7"),
            kwargs.get("Image 7 = 8"),
        ]
        
        if 1 <= select <= 8:
            img_out = images[select - 1]
        else:
            img_out = images[0]  # Default to first image if selection is out of range
        
        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "StarImageSwitch": StarImageSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageSwitch": "⭐ Star Input Image Chooser"
}