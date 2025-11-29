import math
import torch


class StarAdvanvesRatioLatent:
    # Base area for "1 MP" in this node: SDXL-style 1024x1024 (~1.05M pixels)
    BASE_AREA = 1024 * 1024.0
    @classmethod
    def INPUT_TYPES(cls):
        ratios = [
            ("custom", None),
            ("1:1", (1, 1)),
            ("3:4", (3, 4)),
            ("5:7", (5, 7)),
            ("9:16", (9, 16)),
            ("4:3", (4, 3)),
            ("7:5", (7, 5)),
            ("16:9", (16, 9)),
        ]

        ratio_labels = [r[0] for r in ratios]
        ratio_map = {r[0]: r[1] for r in ratios}

        # Store mapping on the class for reuse in create
        cls._RATIO_MAP = ratio_map

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

        latent_channels_options = [4, 16]

        return {
            "required": {
                "ratio": (ratio_labels, {"default": "16:9"}),
                "custom_ratio": ("STRING", {"default": "21:9", "multiline": False, "placeholder": "e.g. 21:9"}),
                "resolution": (resolution_options, {"default": "4 Megapixel (≈ 2000x2000)"}),
                "custom_width": ("INT", {"default": 1920, "min": 16, "max": 99968, "step": 16}),
                "custom_height": ("INT", {"default": 1080, "min": 16, "max": 99968, "step": 16}),
                "latent_channels": (latent_channels_options, {"default": 16}),
                "ratio_from_image": ("BOOLEAN", {"default": False, "display_name": "Ratio From Image"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT")
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Image And Latent"

    @staticmethod
    def _round_to_multiple(value: float, multiple: int = 16) -> int:
        return int(max(multiple, round(value / multiple) * multiple))

    @classmethod
    def _get_ar_from_label(cls, label: str):
        mapping = getattr(cls, "_RATIO_MAP", None)
        if not mapping:
            return None
        return mapping.get(label)

    @staticmethod
    def _aspect_from_image(image):
        try:
            if image is None:
                return None
            # image can be [H, W, C] or [B, H, W, C]
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
    def create(
        cls,
        ratio: str,
        custom_ratio: str,
        resolution: str,
        custom_width: int,
        custom_height: int,
        latent_channels: int,
        ratio_from_image: bool = False,
        image=None,
        batch_size: int = 1,
    ):
        # Determine aspect ratio
        ar_w, ar_h = 16, 9  # default

        if ratio_from_image and image is not None:
            target_ar = cls._aspect_from_image(image)
            if target_ar is not None:
                # Find nearest predefined ratio by aspect
                candidates = [v for v in cls._RATIO_MAP.values() if v is not None]
                best = None
                best_err = 1e9
                for rw, rh in candidates:
                    ar = rw / rh
                    err = abs(ar - target_ar)
                    if err < best_err:
                        best_err, best = err, (rw, rh)
                if best is not None:
                    ar_w, ar_h = best
        else:
            # If user selected a preset ratio (non-custom), use that
            chosen = cls._get_ar_from_label(ratio)
            if chosen is not None:
                ar_w, ar_h = chosen
            else:
                # Handle custom text ratio like "21:9" or "4:3" when ratio == "custom"
                ar_w, ar_h = 1, 1
                try:
                    txt = (custom_ratio or "").strip()
                    if txt:
                        # allow formats like "21:9" or "21x9"
                        if ":" in txt:
                            parts = txt.split(":", 1)
                        elif "x" in txt.lower():
                            parts = txt.lower().split("x", 1)
                        else:
                            parts = []
                        if len(parts) == 2:
                            w_val = int(parts[0].strip())
                            h_val = int(parts[1].strip())
                            if w_val > 0 and h_val > 0:
                                ar_w, ar_h = w_val, h_val
                except Exception:
                    # If anything fails, keep fallback 1:1
                    ar_w, ar_h = 1, 1

        # If resolution is custom, use custom width/height and ignore ratio/area logic
        if resolution == "custom":
            width = cls._round_to_multiple(custom_width, 16)
            height = cls._round_to_multiple(custom_height, 16)

            width = max(16, width)
            height = max(16, height)

            width_latent = max(2, width // 8)
            height_latent = max(2, height // 8)

            latent = torch.zeros([batch_size, int(latent_channels), int(height_latent), int(width_latent)])
            return ({"samples": latent}, int(width), int(height))

        # Clean label so logic is robust to extra info in parentheses, e.g. "SD (512x512)"
        base_resolution = resolution.split(" (", 1)[0]

        # Handle special resolution presets first
        special_areas = {
            "Qwen Image": 1328 * 1328,       # ~1.76M pixels
            "WAN HD": 1280 * 720,           # ~0.92M pixels
            "WAN FullHD": 1920 * 1080,      # ~2.07M pixels (will still be snapped to /16)
        }

        if base_resolution in special_areas:
            target_area = float(special_areas[base_resolution])
        else:
            # Parse numeric megapixels (or SDXL alias) as a scale against BASE_AREA
            try:
                if base_resolution == "SDXL":
                    mp_value = 1.0
                elif base_resolution == "SD":
                    # SD-style 512x512 base area relative to 1024x1024 BASE_AREA
                    mp_value = (512 * 512) / cls.BASE_AREA
                else:
                    # Expect labels like "4 Megapixel" -> extract leading number
                    txt = base_resolution.split(" ")[0]
                    mp_value = float(txt)
            except Exception:
                mp_value = 4.0
            mp_value = max(0.1, mp_value)

            # Total pixels (image space), scaled relative to a 1024x1024 base area
            # (this is more in line with diffusion model training resolutions)
            target_area = mp_value * cls.BASE_AREA

        # Compute ideal float width/height from area and aspect
        ideal_w = math.sqrt(target_area * (ar_w / ar_h))
        ideal_h = ideal_w * (ar_h / ar_w)

        # Snap to multiples of 16, with a minimum of 16
        width = cls._round_to_multiple(ideal_w, 16)
        height = cls._round_to_multiple(ideal_h, 16)

        # Ensure strictly positive
        width = max(16, width)
        height = max(16, height)

        # Latent resolution: standard factor 8 downscale from image
        width_latent = max(2, width // 8)
        height_latent = max(2, height // 8)

        latent = torch.zeros([batch_size, int(latent_channels), int(height_latent), int(width_latent)])
        return ({"samples": latent}, int(width), int(height))


NODE_CLASS_MAPPINGS = {
    "StarAdvanvesRatioLatent": StarAdvanvesRatioLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarAdvanvesRatioLatent": "⭐ Star Advanved Ratio/Latent",
}
