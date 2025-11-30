import math

import torch
import torch.nn.functional as F


class StarLatentResize:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"    # Title color
    CATEGORY = "⭐StarNodes/Image And Latent"
    BASE_AREA = 1024 * 1024.0

    @classmethod
    def INPUT_TYPES(cls):
        # Two modes: keep aspect from input latent, or use a custom ratio string
        ratio_labels = [
            "Keep Input Ratio",
            "Custom Size",
        ]

        resolution_options = [
            "custom",                                  # direct width/height
            "SD (≈ 512x512)",                          # ~512x512 base
            "SDXL (≈ 1024x1024)",                      # ~1024x1024 base
            "Qwen Image (≈ 1328x1328)",                # ~1328x1328
            "WAN HD (≈ 1280x720)",                     # ~1280x720
            "2 Megapixel (≈ 1408x1408)",
            "WAN FullHD (≈ 1920x1080)",                # ~1920x1080
            "3 Megapixel (≈ 1728x1728)",
            "4 Megapixel (≈ 2000x2000)",
            "5 Megapixel (≈ 2240x2240)",
            "6 Megapixel (≈ 2448x2448)",
            "7 Megapixel (≈ 2640x2640)",
            "8 Megapixel (≈ 2832x2832)",
            "9 Megapixel (≈ 3008x3008)",
            "10 Megapixel (≈ 3168x3168)",
            "11 Megapixel (≈ 3328x3328)",
            "12 Megapixel (≈ 3456x3456)",
            "15 Megapixel (≈ 3888x3888)",
            "20 Megapixel (≈ 4480x4480)",
            "50 Megapixel (≈ 7088x7088)",
            "100 Megapixel (≈ 10000x10000)",
        ]

        return {
            "required": {
                "LATENT": ("LATENT",),
                "ratio": (ratio_labels, {"default": "Keep Input Ratio"}),
                "resolution": (resolution_options, {"default": "2 Megapixel (≈ 1408x1408)"}),
                "custom_width": ("INT", {"default": 1920, "min": 16, "max": 99968, "step": 16}),
                "custom_height": ("INT", {"default": 1080, "min": 16, "max": 99968, "step": 16}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT")
    FUNCTION = "resize"
    CATEGORY = CATEGORY

    @staticmethod
    def _round_to_multiple(value: float, multiple: int = 16) -> int:
        return int(max(multiple, round(value / multiple) * multiple))

    @classmethod
    def _get_ar_from_label(cls, label: str):
        mapping = getattr(cls, "_RATIO_MAP", None)
        if not mapping:
            return None
        return mapping.get(label)

    @classmethod
    def resize(
        cls,
        LATENT,
        ratio: str,
        resolution: str,
        custom_width: int,
        custom_height: int,
    ):
        # Determine target aspect ratio
        # Option 1: keep aspect ratio from input latent
        # Option 2: fall back to 1:1 (for Custom Size or missing latent)
        ar_w, ar_h = 1.0, 1.0

        # Try to read aspect from input latent when requested
        samples = None
        try:
            if LATENT is not None and isinstance(LATENT, dict):
                samples = LATENT.get("samples", None)
        except Exception:
            samples = None

        if ratio == "Keep Input Ratio" and samples is not None and isinstance(samples, torch.Tensor) and samples.ndim == 4:
            _, _, h_lat, w_lat = samples.shape
            if h_lat > 0 and w_lat > 0:
                ar_w, ar_h = float(w_lat), float(h_lat)

        # If resolution is custom, use provided width/height and ignore ratio logic
        if resolution == "custom":
            width = cls._round_to_multiple(custom_width, 16)
            height = cls._round_to_multiple(custom_height, 16)

            width = max(16, width)
            height = max(16, height)
        else:
            base_resolution = resolution.split(" (", 1)[0]

            special_areas = {
                "Qwen Image": 1328 * 1328,
                "WAN HD": 1280 * 720,
                "WAN FullHD": 1920 * 1080,
            }

            if base_resolution in special_areas:
                target_area = float(special_areas[base_resolution])
            else:
                try:
                    if base_resolution == "SDXL":
                        mp_value = 1.0
                    elif base_resolution == "SD":
                        mp_value = (512 * 512) / cls.BASE_AREA
                    else:
                        txt = base_resolution.split(" ")[0]
                        mp_value = float(txt)
                except Exception:
                    mp_value = 4.0
                mp_value = max(0.1, mp_value)
                target_area = mp_value * cls.BASE_AREA

            ideal_w = math.sqrt(target_area * (ar_w / ar_h))
            ideal_h = ideal_w * (ar_h / ar_w)

            width = cls._round_to_multiple(ideal_w, 16)
            height = cls._round_to_multiple(ideal_h, 16)

            width = max(16, width)
            height = max(16, height)

        # Compute target latent spatial size (standard 8x downscale from image)
        target_h_latent = max(1, height // 8)
        target_w_latent = max(1, width // 8)

        samples = None
        try:
            if LATENT is not None and isinstance(LATENT, dict):
                samples = LATENT.get("samples", None)
        except Exception:
            samples = None

        if samples is not None and isinstance(samples, torch.Tensor) and samples.ndim == 4:
            b, c, _, _ = samples.shape
            resized = F.interpolate(samples, size=(target_h_latent, target_w_latent), mode="bilinear", align_corners=False)
        else:
            # Fallback: create new latent if input is missing or invalid
            b, c = 1, 4
            resized = torch.zeros((b, c, target_h_latent, target_w_latent))

        out_latent = {"samples": resized}
        return (out_latent, int(width), int(height))


NODE_CLASS_MAPPINGS = {
    "StarLatentResize": StarLatentResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLatentResize": "⭐ Star Latent Resize",
}
