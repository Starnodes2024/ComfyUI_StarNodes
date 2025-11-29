import torch
import torch.nn.functional as F


class StarBlackWhite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "preset": ([
                    "Neutral Luminance",
                    "Skin High Red",
                    "Sky High Blue",
                    "Green Filter",
                    "High Contrast",
                    "Soft Matte",
                    "Film Grain Subtle",
                    "Film Grain Strong",
                ], {"default": "Neutral Luminance"}),
                "red_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "green_weight": ("FLOAT", {"default": 0.59, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blue_weight": ("FLOAT", {"default": 0.11, "min": 0.0, "max": 2.0, "step": 0.01}),
                "normalize_weights": ("BOOLEAN", {"default": True}),
                "add_grain": ("BOOLEAN", {"default": False}),
                "grain_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_density": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_black_white"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def _generate_grain(self, shape, strength, density, size, device, dtype):
        b, c, h, w = shape
        # Base noise resolution (larger size => coarser grain)
        gh = max(1, h // size)
        gw = max(1, w // size)

        # Uniform noise in [-1, 1]
        base_noise = torch.rand(b, 1, gh, gw, device=device, dtype=dtype) * 2.0 - 1.0

        # Density: mix towards 0 (less visible noise)
        if density < 1.0:
            base_noise = base_noise * density

        # Upsample to image size
        noise = F.interpolate(base_noise, size=(h, w), mode="bilinear", align_corners=False)
        noise = noise.expand(b, c, h, w)

        # Scale by strength
        return noise * strength

    def apply_black_white(
        self,
        image,
        preset: str,
        red_weight: float,
        green_weight: float,
        blue_weight: float,
        normalize_weights: bool,
        add_grain: bool,
        grain_strength: float,
        grain_density: float,
        grain_size: int,
    ):
        # IMAGE is NHWC float in [0,1]
        if not isinstance(image, torch.Tensor):
            return (image,)

        x = image.movedim(-1, 1)  # NHWC -> NCHW
        b, c, h, w = x.shape
        if c < 3:
            # Not RGB, just return as-is
            out = image
            return (out,)

        # Apply preset values; user can then tweak sliders further
        if preset == "Neutral Luminance":
            red_weight = 0.3
            green_weight = 0.59
            blue_weight = 0.11
            normalize_weights = True
        elif preset == "Skin High Red":
            red_weight = 0.7
            green_weight = 0.4
            blue_weight = 0.1
            normalize_weights = True
        elif preset == "Sky High Blue":
            red_weight = 0.2
            green_weight = 0.4
            blue_weight = 0.8
            normalize_weights = True
        elif preset == "Green Filter":
            red_weight = 0.2
            green_weight = 0.8
            blue_weight = 0.1
            normalize_weights = True
        elif preset == "High Contrast":
            red_weight = 0.5
            green_weight = 0.5
            blue_weight = 0.2
            normalize_weights = True
        elif preset == "Soft Matte":
            red_weight = 0.35
            green_weight = 0.55
            blue_weight = 0.1
            normalize_weights = True
        elif preset == "Film Grain Subtle":
            red_weight = 0.3
            green_weight = 0.59
            blue_weight = 0.11
            normalize_weights = True
            grain_strength = 0.15
            grain_density = 0.6
            grain_size = 3
        elif preset == "Film Grain Strong":
            red_weight = 0.3
            green_weight = 0.55
            blue_weight = 0.15
            normalize_weights = True
            grain_strength = 0.35
            grain_density = 0.9
            grain_size = 6

        r_w = float(max(0.0, red_weight))
        g_w = float(max(0.0, green_weight))
        b_w = float(max(0.0, blue_weight))

        weights = torch.tensor([r_w, g_w, b_w], device=x.device, dtype=x.dtype)
        if normalize_weights:
            s = weights.sum()
            if s > 0:
                weights = weights / s

        # Apply channel mix to first 3 channels
        rgb = x[:, :3, :, :]
        # (B,3,H,W) * (3,) -> (B,H,W)
        bw = (rgb * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

        # Expand to 3-channel grayscale for IMAGE
        bw_3 = bw.repeat(1, 3, 1, 1)

        if add_grain and grain_strength > 0.0:
            g_strength = float(max(0.0, min(grain_strength, 1.0)))
            g_density = float(max(0.0, min(grain_density, 1.0)))
            g_size = int(max(1, min(grain_size, 64)))
            noise = self._generate_grain(bw_3.shape, g_strength, g_density, g_size, x.device, x.dtype)
            bw_3 = torch.clamp(bw_3 + noise, 0.0, 1.0)

        out = bw_3.movedim(1, -1)  # NCHW -> NHWC
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StarBlackWhite": StarBlackWhite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarBlackWhite": "⭐ Star Black & White",
}
