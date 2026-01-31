import math
import comfy.utils
import node_helpers
from comfy_extras.nodes_flux import FluxGuidance


class StarFlux2Condition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "text": ("STRING", {"multiline": True, "default": "create a high quality map..."}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "⭐StarNodes/Conditioning"

    def encode(self, clip, vae, text, guidance, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning = [[cond, output]]

        conditioning = FluxGuidance().append(conditioning, guidance)[0]

        images = [image_1, image_2, image_3, image_4, image_5]

        ref_latents = []

        target_pixels = 1000000
        for img in images:
            if img is None:
                continue

            samples = img.movedim(-1, 1)
            total_pixels = int(samples.shape[2] * samples.shape[3])
            if total_pixels > target_pixels:
                scale = math.sqrt(target_pixels / float(total_pixels))
                width = max(1, int(round(samples.shape[3] * scale)))
                height = max(1, int(round(samples.shape[2] * scale)))
                width = max(8, int(round(width / 8.0)) * 8)
                height = max(8, int(round(height / 8.0)) * 8)
                samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")

            width = int(samples.shape[3])
            height = int(samples.shape[2])
            width = max(8, int(round(width / 8.0)) * 8)
            height = max(8, int(round(height / 8.0)) * 8)
            if width != int(samples.shape[3]) or height != int(samples.shape[2]):
                samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")

            img_scaled = samples.movedim(1, -1)
            latent = vae.encode(img_scaled[:, :, :, :3])

            if latent is not None:
                ref_latents.append(latent)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "StarFlux2Condition": StarFlux2Condition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFlux2Condition": "⭐ Star Flux2 Conditioner",
}
