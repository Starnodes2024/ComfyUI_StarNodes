import math

CATEGORY = '⭐StarNodes/Helpers And Tools'

class StarSizeCalculatorBySide:
    """
    Node that calculates new dimensions based on resizing by longest or shortest side
    while maintaining aspect ratio.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
            },
            "required": {
                "use_input_image": ("BOOLEAN", {"default": True}),
                "target_size": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "resize_by_side": (["Longest", "Shortest"],),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("width_str", "width_int", "height_str", "height_int", "long_side_str", "long_side_int", "short_side_str", "short_side_int")
    FUNCTION = "calculate_size"
    OUTPUT_NODE = False
    CATEGORY = CATEGORY

    def calculate_size(self, target_size, resize_by_side, use_input_image=True, image=None, width=None, height=None):
        # Determine source dimensions
        if use_input_image and image is not None:
            # Get dimensions from image tensor
            if hasattr(image, 'shape'):
                if len(image.shape) == 4:
                    # Batch of images: [batch, height, width, channels]
                    _, h, w, _ = image.shape
                elif len(image.shape) == 3:
                    # Single image: [height, width, channels]
                    h, w, _ = image.shape
                else:
                    # Fallback
                    h, w = image.shape[:2]
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            source_width = int(w)
            source_height = int(h)
        else:
            # Use manual width/height inputs
            if width is None or height is None:
                raise ValueError("When not using input image, both width and height must be provided")
            source_width = width
            source_height = height

        # Calculate aspect ratio
        aspect_ratio = source_width / source_height

        # Determine which side to resize by
        if resize_by_side == "Longest":
            if source_width >= source_height:
                # Width is longest
                new_width = target_size
                new_height = int(round(target_size / aspect_ratio))
            else:
                # Height is longest
                new_height = target_size
                new_width = int(round(target_size * aspect_ratio))
        else:  # Shortest
            if source_width <= source_height:
                # Width is shortest
                new_width = target_size
                new_height = int(round(target_size / aspect_ratio))
            else:
                # Height is shortest
                new_height = target_size
                new_width = int(round(target_size * aspect_ratio))

        # Determine long and short sides
        if new_width >= new_height:
            long_side_int = new_width
            short_side_int = new_height
        else:
            long_side_int = new_height
            short_side_int = new_width

        # Return as both string and int
        width_str = str(new_width)
        height_str = str(new_height)
        long_side_str = str(long_side_int)
        short_side_str = str(short_side_int)

        return (width_str, new_width, height_str, new_height, long_side_str, long_side_int, short_side_str, short_side_int)

NODE_CLASS_MAPPINGS = {
    "Star_Size_Calculator_By_Side": StarSizeCalculatorBySide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Star_Size_Calculator_By_Side": "⭐ Star Size Calculator by Side",
}
