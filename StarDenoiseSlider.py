import torch

class StarDenoiseSlider:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    CATEGORY = '⭐StarNodes'
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "denoise_str")
    FUNCTION = "get_denoise_value"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise_value": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Denoise strength value (0.1 to 1.0)"
                }),
            }
        }
    
    DESCRIPTION = """⭐ Star Denoise Slider
    
A simple slider for denoise strength values from 0.1 to 1.0.

Input:
- denoise_value: Slider to select denoise strength (0.1 to 1.0)

Outputs:
- denoise: The selected denoise value as a float
- denoise_str: The selected denoise value as a string
"""

    def get_denoise_value(self, denoise_value):
        # Round to nearest 0.05 step
        steps = round(denoise_value * 20) / 20
        
        # Ensure value is within range
        value = max(0.1, min(1.0, steps))
        
        # Convert to string
        value_str = str(value)
        
        return (value, value_str)

# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "StarDenoiseSlider": StarDenoiseSlider
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarDenoiseSlider": "⭐ Star Denoise Slider"
}
