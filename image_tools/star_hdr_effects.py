import torch
import torch.nn.functional as F


class StarHDREffects:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "preset": ([
                    "Natural HDR",
                    "Strong HDR",
                    "Soft HDR Matte",
                    "Detail Boost",
                    "Sky Protect Highlights",
                ], {"default": "Natural HDR"}),
                # Overall intensity of the HDR effect
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                # Radius for local contrast (bigger = more "HDR halo")
                "radius": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                # How strong to apply local detail/contrast
                "local_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # Global contrast adjustment (1.0 = none)
                "global_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # Color saturation multiplier
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # Protect highlights from clipping
                "highlight_protection": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_hdr"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def _gaussian_kernel(self, radius: int, device, dtype):
        sigma = max(float(radius) / 2.0, 0.1)
        size = radius * 2 + 1
        coords = torch.arange(size, device=device, dtype=dtype) - radius
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def _blur(self, x, radius: int):
        if radius <= 0:
            return x
        b, c, h, w = x.shape
        kernel = self._gaussian_kernel(radius, x.device, x.dtype)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(c, 1, 1, 1)
        padding = radius
        return F.conv2d(x, kernel, padding=padding, groups=c)

    def _adjust_saturation(self, x, saturation: float):
        # x: B,C,H,W in [0,1]
        if x.shape[1] < 3:
            return x
        rgb = x[:, :3, :, :]
        # Luma weights
        luma = (rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114)
        return torch.clamp(luma + (rgb - luma) * saturation, 0.0, 1.0)

    def apply_hdr(
        self,
        image,
        preset: str,
        strength: float,
        radius: int,
        local_contrast: float,
        global_contrast: float,
        saturation: float,
        highlight_protection: float,
    ):
        # IMAGE is NHWC float in [0,1]
        if not isinstance(image, torch.Tensor):
            return (image,)

        x = image.movedim(-1, 1)  # NHWC -> NCHW
        b, c, h, w = x.shape

        # Apply preset values; user can then tweak sliders further
        if preset == "Natural HDR":
            strength = 0.6
            radius = 10
            local_contrast = 1.0
            global_contrast = 1.0
            saturation = 1.0
            highlight_protection = 0.5
        elif preset == "Strong HDR":
            strength = 1.0
            radius = 12
            local_contrast = 1.8
            global_contrast = 1.1
            saturation = 1.1
            highlight_protection = 0.6
        elif preset == "Soft HDR Matte":
            strength = 0.5
            radius = 8
            local_contrast = 0.9
            global_contrast = 0.8
            saturation = 0.9
            highlight_protection = 0.7
        elif preset == "Detail Boost":
            strength = 0.8
            radius = 6
            local_contrast = 2.0
            global_contrast = 1.0
            saturation = 1.0
            highlight_protection = 0.4
        elif preset == "Sky Protect Highlights":
            strength = 0.7
            radius = 10
            local_contrast = 1.2
            global_contrast = 0.95
            saturation = 0.95
            highlight_protection = 0.9

        strength = float(max(0.0, min(strength, 2.0)))
        radius = int(max(1, min(radius, 64)))
        local_contrast = float(max(0.0, min(local_contrast, 3.0)))
        global_contrast = float(max(0.0, min(global_contrast, 3.0)))
        saturation = float(max(0.0, min(saturation, 3.0)))
        highlight_protection = float(max(0.0, min(highlight_protection, 1.0)))

        if strength == 0.0:
            out = image
            return (out,)

        # Base tone map in log space to get HDR-style compression
        eps = 1e-4
        base = self._blur(x, radius)
        # Avoid division by zero
        base = torch.clamp(base, eps, 1.0)

        # Detail layer in log domain
        log_x = torch.log(torch.clamp(x, eps, 1.0))
        log_base = torch.log(base)
        detail = log_x - log_base

        # Enhance detail
        detail_enhanced = detail * local_contrast

        # Reconstruct with detail and global contrast
        log_recon = log_base * global_contrast + detail_enhanced
        recon = torch.exp(log_recon)

        # Blend with original based on overall strength
        y = torch.lerp(x, recon, strength)

        # Simple highlight protection: compress values near 1.0
        if highlight_protection > 0.0:
            hp = highlight_protection
            # Smooth roll-off: y' = 1 - (1 - y)^(1+hp)
            y = 1.0 - (1.0 - torch.clamp(y, 0.0, 1.0)) ** (1.0 + hp)

        # Saturation adjustment in RGB
        y = torch.clamp(y, 0.0, 1.0)
        y = self._adjust_saturation(y, saturation)

        out = y.movedim(1, -1)  # NCHW -> NHWC
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StarHDREffects": StarHDREffects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarHDREffects": "⭐ Star HDR Effects",
}
