# divisibledimensions.py

import torch
import math

class DivisibleDimensions:
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
    FUNCTION = "adjust_dimensions"
    CATEGORY = "dimensions"

    def adjust_dimensions(self, width, height, divisible_by):
        def round_to_nearest_multiple(value, multiple):
            return multiple * round(value / multiple)

        new_width = round_to_nearest_multiple(width, divisible_by)
        new_height = round_to_nearest_multiple(height, divisible_by)

        print(f"Original dimensions: {width}x{height}")
        print(f"New dimensions: {new_width}x{new_height}")

        return (new_width, new_height)
