# divisibledimensions.py

class StarDivisibleDimension:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "adjust_dimensions"  # Changed from "Math" to "adjust_dimensions"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def adjust_dimensions(self, width, height, divisible_by):
        def round_to_nearest_multiple(value, multiple):
            return multiple * round(value / multiple)

        new_width = round_to_nearest_multiple(width, divisible_by)
        new_height = round_to_nearest_multiple(height, divisible_by)

        return (new_width, new_height)
    
NODE_CLASS_MAPPINGS = {
    "StarDivisibleDimension" : StarDivisibleDimension
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarDivisibleDimension": "⭐ Star Divisible Dimension"
}
