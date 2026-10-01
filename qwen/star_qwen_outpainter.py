import math
import os
import json
import numpy as np
import torch
import node_helpers
import comfy.utils
import comfy.samplers
import comfy.model_management
import folder_paths
from PIL import Image, ImageOps
from nodes import common_ksampler
from ..misc.star_preview import apply_star_preview

CATEGORY = "⭐StarNodes/Sampler"
SDRATIOS_PATH = os.path.join(os.path.dirname(__file__), '..', 'json', 'sdratios.json')

MEGAPIXEL_CHOICES = [str(mp) for mp in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]]

DEFAULT_PROMPT = 'expand the red areas with background and scene that fits the source <image 1>. add a fluffy purple otter with a golden star and the word "STARNODES" to the scene.'

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _list_images(directory):
    try:
        return sorted(
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(_IMAGE_EXTS)
        )
    except FileNotFoundError:
        return []


class StarQwenOutpainter:
    """
    All-in-one outpainting for Qwen-Image-Edit models. Loads an image directly
    (input/output folder, upload or clipboard paste), places it on a solid red
    canvas in the chosen aspect ratio and lets the edit model fill the red
    areas. The canvas is the only reference image.
    """
    COLOR = "#19124d"  # Title color
    BGCOLOR = "#3d124d"  # Background color

    @classmethod
    def INPUT_TYPES(cls):
        with open(SDRATIOS_PATH, 'r', encoding='utf-8') as f:
            ratios_data = json.load(f)["ratios"]
        ratio_choices = [k for k in ratios_data.keys() if k != "Free Ratio"]

        files = (_list_images(folder_paths.get_input_directory())
                 + [f"{f} [output]" for f in _list_images(folder_paths.get_output_directory())])
        return {"required": {
                    "model": ("MODEL", ),
                    "clip": ("CLIP", ),
                    "vae": ("VAE", ),
                    "image": (files, {"image_upload": True, "tooltip": "Image to outpaint — from the input folder, the output folder ([output] entries), or upload / paste from clipboard."}),
                    "aspect_ratio": (ratio_choices, {"default": "16:9 [1344x768 landscape]"}),
                    "megapixel": (MEGAPIXEL_CHOICES, {"default": "2.0"}),
                    "image_placement": (["top left", "top center", "top right", "center left", "center", "center right", "bottom left", "bottom center", "bottom right", "custom"], {"default": "center", "tooltip": "Where the input image sits on the new canvas — the rest is filled red and gets outpainted. 'custom' lets you drag the image inside the preview."}),
                    "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": DEFAULT_PROMPT}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "qwen_image_2_1": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No", "tooltip": "Qwen-Image 2.1 mode: 64-channel /16x latents, RGBA VAE output and image_slots conditioning (reference latents on positive and negative)."}),
                    "custom_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Horizontal position for 'custom' placement (0 = left edge, 1 = right edge). Set by dragging the preview."}),
                    "custom_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vertical position for 'custom' placement (0 = top edge, 1 = bottom edge). Set by dragging the preview."}),
                    "custom_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Source image size for 'custom' placement (1.0 = fit canvas). Set by dragging the corner handle in the preview."}),
                },
                "optional": {
                    "reference_image": ("IMAGE", {"tooltip": "Optional second reference image - added to the vision tokens and reference latents next to the red canvas (e.g. a style or content reference)."}),
                    "preview": ("STAR_PREVIEW", {"tooltip": "Optional ⭐ Star Preview options - shows a live sampling preview on the connected ⭐ Star Preview node."}),
                },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("input_image", "image", "reference")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    @staticmethod
    def _resize(img, width, height):
        # img: [B,H,W,C] -> resize keeping width/height order of common_upscale
        return comfy.utils.common_upscale(img.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)

    def build_canvas(self, image, aspect_ratio, megapixel, snap, image_placement, custom_x, custom_y, custom_scale):
        target_pixels = float(megapixel) * 1000000

        img = image[0:1] if image.ndim == 4 else image.unsqueeze(0)
        img = img[:, :, :, :3]
        img_h, img_w = int(img.shape[1]), int(img.shape[2])

        # Cap the input image at the target megapixel size
        if img_w * img_h > target_pixels:
            scale = math.sqrt(target_pixels / (img_w * img_h))
            img_w = max(snap, int(img_w * scale) // snap * snap)
            img_h = max(snap, int(img_h * scale) // snap * snap)
            img = self._resize(img, img_w, img_h)

        # In custom mode the image can also be scaled down freely on the canvas
        if image_placement == "custom" and custom_scale < 1.0:
            img_w = max(snap, int(img_w * custom_scale) // snap * snap)
            img_h = max(snap, int(img_h * custom_scale) // snap * snap)
            img = self._resize(img, img_w, img_h)

        # Base canvas size from the chosen ratio at the chosen megapixel size
        with open(SDRATIOS_PATH, 'r', encoding='utf-8') as f:
            ratios_data = json.load(f)["ratios"]
        base_w = int(ratios_data[aspect_ratio]["width"])
        base_h = int(ratios_data[aspect_ratio]["height"])

        if float(megapixel) == 1.0:
            canvas_w, canvas_h = base_w, base_h
        else:
            aspect = base_w / base_h
            canvas_w = int(math.sqrt(target_pixels * aspect))
            canvas_h = int(canvas_w / aspect)
        canvas_w -= canvas_w % snap
        canvas_h -= canvas_h % snap

        # Grow the canvas (keeping the ratio) until the image fits inside
        fit = max(img_w / canvas_w, img_h / canvas_h)
        if fit > 1.0:
            canvas_w = math.ceil(canvas_w * fit / snap) * snap
            canvas_h = math.ceil(canvas_h * fit / snap) * snap

        # Position of the image on the canvas; the slack is the free red space
        slack_w = canvas_w - img_w
        slack_h = canvas_h - img_h
        place_frac = {
            "top left": (0.0, 0.0), "top center": (0.5, 0.0), "top right": (1.0, 0.0),
            "center left": (0.0, 0.5), "center": (0.5, 0.5), "center right": (1.0, 0.5),
            "bottom left": (0.0, 1.0), "bottom center": (0.5, 1.0), "bottom right": (1.0, 1.0),
        }
        if image_placement == "custom":
            fx = min(max(custom_x, 0.0), 1.0)
            fy = min(max(custom_y, 0.0), 1.0)
        else:
            fx, fy = place_frac.get(image_placement, (0.5, 0.5))
        left = round(slack_w * fx)
        top = round(slack_h * fy)

        # Solid red canvas with the input image placed on it
        canvas = torch.zeros((1, canvas_h, canvas_w, 3), dtype=img.dtype, device=img.device)
        canvas[:, :, :, 0] = 1.0
        canvas[:, top:top + img_h, left:left + img_w, :] = img

        return canvas, canvas_w, canvas_h

    def execute(self, model, clip, vae, image, aspect_ratio, megapixel, image_placement, prompt, seed, steps, cfg, sampler_name, scheduler, denoise, qwen_image_2_1=False, custom_x=0.5, custom_y=0.5, custom_scale=1.0, reference_image=None, preview=None):
        if preview is not None:
            model = apply_star_preview(model, preview)

        if not prompt.strip():
            prompt = DEFAULT_PROMPT

        img_pil = Image.open(folder_paths.get_annotated_filepath(image))
        img_pil = ImageOps.exif_transpose(img_pil)
        img_tensor = torch.from_numpy(np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0)[None,]
        img_pil.close()

        # Qwen 2.1 latents are 64ch at /16, Qwen Image Edit latents are 16ch at /8
        snap = 16 if qwen_image_2_1 else 8
        canvas, canvas_w, canvas_h = self.build_canvas(img_tensor, aspect_ratio, megapixel, snap, image_placement, custom_x, custom_y, custom_scale)

        with torch.no_grad():
            samples = canvas.movedim(-1, 1)

            # Optional second reference, appended after the red canvas
            ref2 = None
            if reference_image is not None:
                ref2 = (reference_image[0:1] if reference_image.ndim == 4
                        else reference_image.unsqueeze(0))[:, :, :, :3]

            if qwen_image_2_1:
                # Same resize for the vision tokens and the VAE reference (multiples of 32, ~1MP)
                ratio = samples.shape[3] / samples.shape[2]
                ref_w = max(32, round(math.sqrt(1024 * 1024 * ratio) / 32) * 32)
                ref_h = max(32, round(math.sqrt(1024 * 1024 / ratio) / 32) * 32)
                ref_img = comfy.utils.common_upscale(samples, ref_w, ref_h, "lanczos", "disabled").movedim(1, -1)
                images_vl = [ref_img[:, :, :, :3]]
                ref_latents = [vae.encode(ref_img)]
                if ref2 is not None:
                    r2 = ref2.movedim(-1, 1)
                    ratio2 = r2.shape[3] / r2.shape[2]
                    ref2_w = max(32, round(math.sqrt(1024 * 1024 * ratio2) / 32) * 32)
                    ref2_h = max(32, round(math.sqrt(1024 * 1024 / ratio2) / 32) * 32)
                    ref2_img = comfy.utils.common_upscale(r2, ref2_w, ref2_h, "lanczos", "disabled").movedim(1, -1)
                    images_vl.append(ref2_img)
                    ref_latents.append(vae.encode(ref2_img))
                conditioning_pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl, keep_vision=False, prevent_empty_text=True))
                conditioning_neg = clip.encode_from_tokens_scheduled(clip.tokenize("", images=images_vl, keep_vision=False, prevent_empty_text=True))
                conditioning_pos = node_helpers.conditioning_set_values(conditioning_pos, {"reference_latents": ref_latents}, append=True)
                conditioning_neg = node_helpers.conditioning_set_values(conditioning_neg, {"reference_latents": ref_latents}, append=True)
            else:
                # TextEncodeQwenImageEdit path: reference image scaled to ~1MP for both
                # the vision tokens and the reference latent
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                ref_w = round(samples.shape[3] * scale_by)
                ref_h = round(samples.shape[2] * scale_by)
                ref_img = comfy.utils.common_upscale(samples, ref_w, ref_h, "area", "disabled").movedim(1, -1)
                images_vl = [ref_img[:, :, :, :3]]
                ref_latents = [vae.encode(ref_img[:, :, :, :3])]
                if ref2 is not None:
                    r2 = ref2.movedim(-1, 1)
                    scale_by2 = math.sqrt(total / (r2.shape[3] * r2.shape[2]))
                    ref2_img = comfy.utils.common_upscale(r2, round(r2.shape[3] * scale_by2), round(r2.shape[2] * scale_by2), "area", "disabled").movedim(1, -1)
                    images_vl.append(ref2_img)
                    ref_latents.append(vae.encode(ref2_img))
                conditioning_pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl))
                conditioning_neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
                conditioning_pos = node_helpers.conditioning_set_values(conditioning_pos, {"reference_latents": ref_latents}, append=True)

            downscale = vae.downscale_ratio if isinstance(vae.downscale_ratio, int) else 8
            latent = torch.zeros([1, vae.latent_channels, canvas_h // downscale, canvas_w // downscale], device=comfy.model_management.intermediate_device())

            latent_result = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                    conditioning_pos, conditioning_neg, {"samples": latent}, denoise=denoise)[0]

            decoded_image = vae.decode(latent_result["samples"])
            if len(decoded_image.shape) == 5: # video-style VAEs return [B, T, H, W, C]
                decoded_image = decoded_image.reshape(-1, decoded_image.shape[-3], decoded_image.shape[-2], decoded_image.shape[-1])
            if decoded_image.shape[-1] > 3: # the Qwen Image 2.1 VAE decodes RGBA, drop the alpha channel
                decoded_image = decoded_image[:, :, :, :3]

        torch.cuda.empty_cache()

        return (img_tensor, decoded_image, canvas)

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            return f"{os.path.getmtime(image_path)}-{os.path.getsize(image_path)}"
        except OSError:
            return image

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarQwenOutpainter": StarQwenOutpainter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenOutpainter": "⭐ Star Qwen2 Outpainter",
}
