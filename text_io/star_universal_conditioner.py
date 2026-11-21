"""Star Universal Conditioner

One conditioning node for every image and image-edit model. It encodes the
positive and negative prompt with the connected CLIP and feeds the connected
reference images (up to ten fixed inputs) the way each model family
expects:

- vision-language text encoders (Qwen Image / 2.1, Krea 2, MOSS-Image,
  Fibo2, Flux 3, Boogu, JoyImage, MageFlow, LTXV, ...) get the images inside
  tokenize(), so their vision tower sees them;
- when a VAE is connected the images are also encoded and appended as
  "reference_latents" to both conditionings - the standard key consumed by
  Flux/Flux2/Flux3, Qwen Image, Krea2, MOSSImage, Fibo2 and every other
  edit model (models that don't use it, like plain SDXL, just ignore it).

"Negative Zero" applies the native ConditioningZeroOut behaviour to the
negative output (zeroed cross-attention and pooled fields, reference
latents kept).
"""

import inspect
import logging
import math

import torch

import comfy.utils
import node_helpers

_MAX_IMAGES = 10

_REF_RESIZE = {
    "none": 0.0,
    "0.5 MP": 0.5,
    "1 MP": 1.0,
    "1.5 MP": 1.5,
    "2 MP": 2.0,
    "3 MP": 3.0,
}


def _supports_images(clip):
    """True when the loaded CLIP's tokenizer accepts image inputs - the
    vision-language encoders (Qwen3VL / Qwen3.5 / Qwen Image / Fibo2 /
    Krea2 / Flux3 / Boogu / ...). Flux2 dev (Mistral3), Klein and SDXL
    style encoders return False."""
    try:
        return "images" in inspect.signature(
            clip.tokenizer.tokenize_with_weights).parameters
    except (TypeError, ValueError):
        return False


def _resize_ref(image, megapixels):
    """Scale an [B,H,W,C] image to about megapixels, snapped to a 32-pixel
    grid (the vision-slot grid the edit models are trained on)."""
    height, width = image.shape[1:3]
    scale = math.sqrt(megapixels * 1024 * 1024 / max(1, width * height))
    width = max(32, round(width * scale / 32) * 32)
    height = max(32, round(height * scale / 32) * 32)
    if (width, height) == (image.shape[2], image.shape[1]):
        return image
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


def _zero_out(conditioning):
    """Same as the native ConditioningZeroOut node: zero the cond tensor and
    the pooled-style fields, keep everything else (e.g. reference_latents)."""
    c = []
    for t in conditioning:
        d = t[1].copy()
        for key in ("pooled_output", "conditioning_lyrics",
                    "conditioning_scale", "direct_context"):
            if d.get(key) is not None:
                d[key] = torch.zeros_like(d[key])
        c.append([torch.zeros_like(t[0]), d])
    return c


class StarUniversalConditioner:
    """Universal prompt + reference-image conditioning for every image and
    image-edit model family (SDXL, Flux/Klein, Flux2/Flux3, Qwen Image,
    Krea 2, MOSS-Image, Fibo2, ...)."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"vae": ("VAE", {
            "tooltip": "Encodes the reference images into the "
                       "reference_latents the edit models consume. "
                       "Without it the images only reach the text "
                       "encoder's vision tower (VL encoders only)."})}
        for i in range(1, _MAX_IMAGES + 1):
            optional[f"Image {i}"] = ("IMAGE", {
                "tooltip": "Reference image for edit models - connect up "
                           "to ten, used in order."})
        return {
            "required": {
                "clip": ("CLIP", {}),
                "positive": ("STRING", {
                    "multiline": True, "dynamicPrompts": True,
                    "tooltip": "Positive prompt."}),
                "negative": ("STRING", {
                    "multiline": True, "dynamicPrompts": True,
                    "tooltip": "Negative prompt - left empty it produces "
                               "each model's own empty/null conditioning."}),
                "negative_zero": ("BOOLEAN", {
                    "default": False, "display_name": "Negative Zero",
                    "tooltip": "Zero the negative conditioning like the "
                               "native ConditioningZeroOut node "
                               "(reference latents are kept)."}),
                "ref_resize": (list(_REF_RESIZE), {
                    "default": "1 MP", "display_name": "Reference Resize",
                    "tooltip": "Size the reference images are scaled to "
                               "before VAE encode / vision tower, keeping "
                               "aspect ratio on a 32 px grid - important "
                               "for image-edit models. 'none' keeps each "
                               "image at its own resolution."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "⭐StarNodes/Conditioning"
    DESCRIPTION = ("Universal prompt conditioning for every image and "
                   "image-edit model: encodes text, feeds connected "
                   "reference images to the vision text encoder and/or as "
                   "VAE reference latents, and can zero the negative.")

    def encode(self, clip, positive, negative, negative_zero, ref_resize,
               vae=None, **kwargs):
        images = [kwargs.get(f"Image {i}") for i in range(1, _MAX_IMAGES + 1)]
        images = [im for im in images if im is not None]

        vision = _supports_images(clip)
        if images and not vision and vae is None:
            logging.warning("[Star Universal Conditioner] %d image(s) "
                            "connected, but the CLIP has no vision input "
                            "and no VAE is connected - images ignored.",
                            len(images))

        ref_megapixels = _REF_RESIZE[ref_resize]
        images_vl = []     # vision-tower inputs, alpha composited on white
        ref_latents = []   # VAE-encoded references for the DiT
        for image in images:
            s = image if ref_megapixels <= 0 else _resize_ref(image, ref_megapixels)
            if vision:
                for b in range(s.shape[0]):
                    frame = s[b:b + 1]
                    rgb = frame[..., :3]
                    if frame.shape[-1] > 3:
                        # the vision tower sees alpha over white, the vae
                        # keeps all four channels (native Qwen 2.1 recipe)
                        rgb = rgb * frame[..., 3:] + (1.0 - frame[..., 3:])
                    images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        tok_kwargs = {}
        if vision and images_vl:
            tok_kwargs = {
                "images": images_vl,
                # qwen image 2.1: vision tokens stay in the text embed only
                # when no reference latents carry the images instead;
                # other tokenizers ignore these kwargs
                "keep_vision": not ref_latents,
                "prevent_empty_text": True,
            }

        positive_cond = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive, **tok_kwargs))
        negative_cond = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative, **tok_kwargs))

        if ref_latents:
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, {"reference_latents": ref_latents}, append=True)
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond, {"reference_latents": ref_latents}, append=True)
            logging.info("[Star Universal Conditioner] %d reference "
                         "latent(s) appended", len(ref_latents))

        if negative_zero:
            negative_cond = _zero_out(negative_cond)

        return (positive_cond, negative_cond)


NODE_CLASS_MAPPINGS = {
    "StarUniversalConditioner": StarUniversalConditioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarUniversalConditioner": "⭐ Star Universal Conditioner",
}
