import math

class StarLTXVideoSettings:
    
    HD_RATIOS = {
        "1:1": (1280, 1280),
        "4:3": (1280, 960),
        "3:2": (1280, 853),
        "16:10": (1280, 800),
        "16:9": (1280, 720),
        "21:9": (1280, 548),
        "3:4": (960, 1280),
        "2:3": (853, 1280),
        "10:16": (800, 1280),
        "9:16": (720, 1280),
        "9:21": (548, 1280),
    }
    
    FHD_RATIOS = {
        "1:1": (1920, 1920),
        "4:3": (1920, 1440),
        "3:2": (1920, 1280),
        "16:10": (1920, 1200),
        "16:9": (1920, 1080),
        "21:9": (1920, 823),
        "3:4": (1440, 1920),
        "2:3": (1280, 1920),
        "10:16": (1200, 1920),
        "9:16": (1080, 1920),
        "9:21": (823, 1920),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        ratio_options = list(cls.HD_RATIOS.keys())
        
        return {
            "required": {
                "video_size": (["HD", "FHD", "Custom"], {"default": "FHD"}),
                "ratio": (ratio_options, {"default": "16:9"}),
                "custom_width": ("INT", {"default": 1920, "min": 32, "max": 8192, "step": 32}),
                "custom_height": ("INT", {"default": 1080, "min": 32, "max": 8192, "step": 32}),
                "best_size_from_input": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "seconds": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "width_float", "height", "height_float", "width_50%", "width_50%_float", "height_50%", "height_50%_float", "frames", "frames_float", "fps", "fps_float", "seconds", "seconds_float")
    FUNCTION = "calculate_settings"
    CATEGORY = "⭐StarNodes/LTX Video"
    
    @staticmethod
    def _round_to_divisible_32_plus_1(value):
        """Round to nearest (multiple of 32) + 1"""
        base = int(round((value - 1) / 32) * 32)
        return max(32, base) + 1
    
    @staticmethod
    def _round_to_divisible_8_plus_1(value):
        """Round to nearest (multiple of 8) + 1"""
        base = int(round((value - 1) / 8) * 8)
        return max(8, base) + 1
    
    @staticmethod
    def _get_image_ratio(image):
        """Extract aspect ratio from input image"""
        if image is None:
            return None
        try:
            if image.ndim == 3:
                h, w = int(image.shape[0]), int(image.shape[1])
            else:
                h, w = int(image.shape[1]), int(image.shape[2])
            if h <= 0 or w <= 0:
                return None
            return w / h
        except Exception:
            return None
    
    @classmethod
    def _find_closest_ratio(cls, target_ratio, size_dict):
        """Find the closest predefined ratio to the target"""
        best_ratio_key = "16:9"
        best_diff = float('inf')
        
        for ratio_key, (w, h) in size_dict.items():
            ratio_value = w / h
            diff = abs(ratio_value - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_ratio_key = ratio_key
        
        return best_ratio_key
    
    def calculate_settings(self, video_size, ratio, custom_width, custom_height, 
                          best_size_from_input, fps, seconds, image=None):
        
        # Determine which ratio dictionary to use
        if video_size == "HD":
            ratio_dict = self.HD_RATIOS
        elif video_size == "FHD":
            ratio_dict = self.FHD_RATIOS
        else:  # Custom
            ratio_dict = None
        
        # Calculate width and height
        if best_size_from_input and image is not None:
            # Get ratio from input image
            image_ratio = self._get_image_ratio(image)
            if image_ratio is not None and ratio_dict is not None:
                # Find closest ratio in the selected size dictionary
                closest_ratio = self._find_closest_ratio(image_ratio, ratio_dict)
                width, height = ratio_dict[closest_ratio]
            elif video_size == "Custom" and image_ratio is not None:
                # For custom size, calculate dimensions based on image ratio
                # Use custom_width as base and calculate height
                width = custom_width
                height = int(custom_width / image_ratio)
            else:
                # Fallback to selected ratio
                if ratio_dict is not None:
                    width, height = ratio_dict[ratio]
                else:
                    width, height = custom_width, custom_height
        elif video_size == "Custom":
            # Use custom dimensions
            width, height = custom_width, custom_height
        else:
            # Use selected ratio from HD or FHD
            width, height = ratio_dict[ratio]
        
        # Ensure width and height are divisible by 32 + 1
        width = self._round_to_divisible_32_plus_1(width)
        height = self._round_to_divisible_32_plus_1(height)
        
        # Calculate 50% dimensions (also divisible by 32 + 1)
        width_50 = self._round_to_divisible_32_plus_1(width / 2)
        height_50 = self._round_to_divisible_32_plus_1(height / 2)
        
        # Calculate frames: fps * seconds + 1, must be divisible by 8 + 1
        frames_raw = fps * seconds + 1
        frames = self._round_to_divisible_8_plus_1(frames_raw)
        
        # Return all outputs as both int and float
        return (
            width, float(width),
            height, float(height),
            width_50, float(width_50),
            height_50, float(height_50),
            frames, float(frames),
            fps, float(fps),
            seconds, float(seconds)
        )


NODE_CLASS_MAPPINGS = {
    "StarLTXVideoSettings": StarLTXVideoSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLTXVideoSettings": "⭐ Star LTX Video Settings",
}
