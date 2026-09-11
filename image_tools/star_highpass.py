import torch
import torch.nn.functional as F


class StarSpecialFilters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_highpass"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def _gaussian_kernel(self, radius: int, device, dtype):
        # Simple 1D Gaussian kernel approximated from radius
        # sigma ~ radius / 2
        sigma = max(float(radius) / 2.0, 0.1)
        size = radius * 2 + 1
        coords = torch.arange(size, device=device, dtype=dtype) - radius
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def apply_highpass(self, image, radius: int, strength: float):
        # IMAGE is NHWC float in [0,1]
        if not isinstance(image, torch.Tensor):
            return (image,)

        x = image.movedim(-1, 1)  # NHWC -> NCHW
        b, c, h, w = x.shape

        radius = int(max(1, min(radius, 64)))
        strength = float(strength)

        kernel = self._gaussian_kernel(radius, x.device, x.dtype)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(c, 1, 1, 1)

        padding = radius
        x_blur = F.conv2d(x, kernel, padding=padding, groups=c)

        highpass = x - x_blur
        # Highpass sharpening: original + strength * highpass
        out = x + strength * highpass
        out = torch.clamp(out, 0.0, 1.0)

        out = out.movedim(1, -1)  # NCHW -> NHWC
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StarSpecialFilters": StarSpecialFilters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSpecialFilters": "⭐ Star HighPass Filters",
}
