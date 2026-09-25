import torch
import torch.nn.functional as F


class StarHighPassOverlay:
    """
    Photoshop-style High Pass sharpening.
    The blurred copy is subtracted onto a neutral grey layer, that layer is
    blended back in Overlay mode, then flattened.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 10, "min": 1, "max": 250, "step": 1,
                                   "tooltip": "High pass radius - controls the detail scale that gets boosted (like the Photoshop radius)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Opacity of the high pass overlay layer, like the Photoshop layer percentage."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def _gaussian_kernel(self, radius: int, device, dtype):
        sigma = max(float(radius) / 2.0, 0.1)
        coords = torch.arange(radius * 2 + 1, device=device, dtype=dtype) - radius
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = gauss[:, None] * gauss[None, :]
        return kernel_2d / kernel_2d.sum()

    def apply(self, image, radius: int, strength: float):
        if not isinstance(image, torch.Tensor):
            return (image,)

        x = image.movedim(-1, 1)  # NHWC -> NCHW
        if x.shape[1] == 4:
            x = x[:, :3]  # RGBA -> RGB, alpha channel dropped

        radius = int(max(1, min(radius, 250)))
        strength = float(max(0.0, min(strength, 1.0)))

        c = x.shape[1]
        kernel = self._gaussian_kernel(radius, x.device, x.dtype)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1]).repeat(c, 1, 1, 1)
        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        blur = F.conv2d(x_pad, kernel, groups=c)

        highpass = (x - blur + 0.5).clamp(0.0, 1.0)
        overlay = torch.where(x <= 0.5, 2.0 * x * highpass, 1.0 - 2.0 * (1.0 - x) * (1.0 - highpass))
        out = (x + (overlay - x) * strength).clamp(0.0, 1.0)

        return (out.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "StarHighPassOverlay": StarHighPassOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarHighPassOverlay": "⭐ Star HighPass Overlay",
}
