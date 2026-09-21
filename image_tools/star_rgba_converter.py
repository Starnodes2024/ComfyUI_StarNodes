import torch


class StarRGBAConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mode": (["RGBA To RGB", "RGB To RGBA"], {"default": "RGBA To RGB"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def convert(self, image, mode):
        channels = image.shape[-1]
        if mode == "RGBA To RGB":
            if channels == 4:
                return (image[:, :, :, :3],)
        else:
            if channels == 3:
                alpha = torch.ones_like(image[:, :, :, :1])
                return (torch.cat([image, alpha], dim=-1),)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "StarRGBAConverter": StarRGBAConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRGBAConverter": "⭐ Star RGBA Converter",
}
