import torch
import torch.nn.functional as F


class StarRadialBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mode": (["Zoom", "Spin"], {"default": "Zoom"}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "samples": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_radial_blur"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def apply_radial_blur(self, image, mode: str, strength: float, samples: int, center_x: float, center_y: float):
        # IMAGE is NHWC float in [0,1]
        if not isinstance(image, torch.Tensor):
            return (image,)

        x = image.movedim(-1, 1)  # NHWC -> NCHW
        b, c, h, w = x.shape

        strength = float(max(0.0, strength))
        samples = int(max(1, min(samples, 64)))

        if strength == 0.0 or samples == 0:
            return (image,)

        # Normalized center in [-1, 1]
        cx = float(center_x) * 2.0 - 1.0
        cy = float(center_y) * 2.0 - 1.0

        # Base grid in normalized coordinates [-1, 1]
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # H, W, 2

        center = torch.tensor([cx, cy], device=x.device, dtype=x.dtype)
        base_grid = base_grid.unsqueeze(0)  # 1, H, W, 2

        acc = x.clone()

        if mode == "Spin":
            # Spin-style radial blur: rotate around center
            rel = base_grid - center  # 1, H, W, 2
            px = rel[..., 0]
            py = rel[..., 1]
            radius = torch.sqrt(px * px + py * py) + 1e-6
            angle = torch.atan2(py, px)

            for i in range(samples):
                t = (i + 1) / float(samples)
                # Angle offset in radians; scale strength to a reasonable range
                dtheta = strength * t * 0.5
                ang = angle + dtheta
                sx = torch.cos(ang) * radius
                sy = torch.sin(ang) * radius
                sample_grid = torch.stack((sx, sy), dim=-1) + center
                sample_grid_b = sample_grid.expand(b, -1, -1, -1)
                sampled = F.grid_sample(x, sample_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
                acc = acc + sampled
        else:
            # Zoom-style radial blur (existing behavior)
            for i in range(samples):
                t = (i + 1) / float(samples)
                # Zoom factor > 1.0
                s = 1.0 + strength * t
                # Move coordinates towards the center for zoom blur
                sample_grid = center + (base_grid - center) / s
                sample_grid_b = sample_grid.expand(b, -1, -1, -1)
                sampled = F.grid_sample(x, sample_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
                acc = acc + sampled

        out = acc / float(samples + 1)
        out = torch.clamp(out, 0.0, 1.0)

        out = out.movedim(1, -1)  # NCHW -> NHWC
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StarRadialBlur": StarRadialBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRadialBlur": "⭐ Star Radial Blur",
}
