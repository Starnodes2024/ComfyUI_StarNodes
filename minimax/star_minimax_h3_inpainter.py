"""Star Minimax H3 Image Inpainter

The ⭐ Star Flux2/Qwen-Image-Edit Inpainter workflow for the MiniMax H3
packed-AV model, in a single node: crop to the mask's context area, encode
the still into the model's single-frame image-mode latent (video member
[1, 24, 1, H/16, W/16] plus the silent two-frame audio member the packed
layout always carries), run a denoise-masked sampling pass and stitch the
decoded still back into the original image.

None of the Flux concat machinery is needed - H3 consumes a per-token
denoise mask natively: mask value m puts a DiT row at sigma = m * sigma,
preserved rows sit at the cond timestep and the clean source latent is
re-injected every step. Soft/blurred masks therefore translate directly
into per-row denoise strength (the model's own differential diffusion).

The source image is presented to the Qwen3-VL text encoder and the DiT as
<Picture 1> - with the masked area blanked to gray (inpaint convention) or
untouched, selectable - and extra reference images become <Picture 2>,
<Picture 3>, ... exactly like the ref2va path of ⭐ Star Minimax All In One.
"""

import logging
import math

import torch
import node_helpers
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from ..misc.StarFlux2Inpainter import StarFlux2Inpainter
from ..misc.star_preview import apply_star_preview
from .minimax_common import decode_still, run_sample
from .star_minimax_aio import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    StarMinimaxAllInOne,
    _resize,
)

MEGAPIXEL_OPTIONS = ["native", 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

DEFAULT_PROMPT = "Seamlessly fill the masked area of <Picture 1>, matching the surrounding content."


def _scale_to_area(image, mask, megapixels):
    """Aspect-preserving scale of the crop + mask to the requested pixel area,
    rounded to the H3 canvas grid (multiple of 32 -> even latent size)."""
    h, w = image.shape[1], image.shape[2]
    scale = math.sqrt(megapixels * 1024 * 1024 / (h * w))
    nw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    nh = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if (nw, nh) == (w, h):
        return image, mask
    img = _resize(image, nw, nh, "disabled")
    m = comfy.utils.common_upscale(
        mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]),
        nw, nh, "bilinear", "disabled").reshape(-1, nh, nw)
    return img, m


def _image_ref(img, vae, ref_image_size, canvas_w, canvas_h):
    """One MiniMaxH3ReferenceToVideo-style image reference: 'match' scales it
    (down only, keeping aspect) to the generation's pixel area, 'max' uses the
    2048px short edge. Returns the (tokenizer item, DiT ref block) pair."""
    h, w = img.shape[1], img.shape[2]
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((canvas_w * canvas_h) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = _resize(img[:1], tw, th, "disabled")
    item = {"type": "image", "data": resized}
    block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
             "latent": vae.encode(resized)}
    return item, block


class StarMinimaxH3ImageInpainter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StarMinimaxH3ImageInpainter",
            display_name="⭐ Star Minimax H3 Image Inpainter",
            category="⭐StarNodes/Sampler",
            description=(
                "Image inpainting with the MiniMax H3 model in its native "
                "single-frame image mode - the ⭐ Star Flux2/Qwen-Image-Edit "
                "Inpainter workflow for H3: the mask's context area is cropped, "
                "VAE-encoded into the packed AV latent and sampled with the "
                "model's native per-token denoise mask (preserved rows are "
                "pinned to the source, masked rows regenerate at mask * sigma, "
                "so soft mask edges blend natively). The decoded still is "
                "stitched back with a blended mask edge. The source image is "
                "presented as <Picture 1> - blanked at the mask (inpaint "
                "convention) or untouched, selectable; extra reference images "
                "become <Picture 2>, <Picture 3>, ... Connect the H3 diffusion "
                "model (UNETLoader, or a Star Minimax AIO MODEL output), the "
                "minimax text encoder (Qwen3-VL) and the H3 video VAE."
            ),
            inputs=[
                # ---------------- Models ----------------
                io.Model.Input("model",
                               tooltip="MiniMax H3 diffusion model (e.g. from UNETLoader or a ⭐ Star Minimax All In One MODEL output)."),
                io.Clip.Input("clip",
                              tooltip="The MiniMax H3 text encoder (Qwen3-VL-32B, clip type 'minimax')."),
                io.Vae.Input("vae",
                             tooltip="The MiniMax H3 video VAE (e.g. minimax_h3_video_vae_fp16)."),
                # ---------------- Inpaint inputs ----------------
                io.Image.Input("image"),
                io.Mask.Input("mask",
                              tooltip="1/white marks the regions to regenerate."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="What to generate in the masked area. The source image is <Picture 1>, extra reference images are <Picture 2>, <Picture 3>, ..."),
                io.Combo.Input("reference", options=["masked source", "source", "off"],
                               default="masked source",
                               tooltip="How the source crop enters the conditioning as <Picture 1>: 'masked source' shows it with the inpaint area blanked to gray (the mask's hole signals what to fill), 'source' shows it untouched (edit style - the model can see what was there), 'off' sends no picture to the text encoder at all."),
                io.Combo.Input("megapixels", options=MEGAPIXEL_OPTIONS, default="native",
                               tooltip="Render resolution of the inpaint crop. 'native' keeps the crop's own pixels; the MP presets rescale crop + mask to that pixel area first - H3 stills hit their quality sweet spot around 3 MP and up. The result is scaled back to the crop size for stitching."),
                # ---------------- Sampling ----------------
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="CFG scale. At 1.0 the negative prompt is skipped and the positive-only basic guider is used (the H3 preset)."),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES,
                               default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES,
                               default="simple",
                               tooltip="'beta' or 'normal' tends to outperform 'simple' for reference-heavy prompts."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.String.Input("negative_prompt", default="",
                                tooltip="Negative prompt - only encoded when cfg differs from 1.0."),
                # ---------------- Mask / crop tuning ----------------
                io.Int.Input("expand_pixels", default=32, min=0, max=512, advanced=True,
                             tooltip="Context margin added around the masked area before sampling - more context helps the model blend the fill into the scene."),
                io.Float.Input("mask_blur", default=16.0, min=0.0, max=256.0, step=0.5, advanced=True,
                               tooltip="Grow + blur applied to the denoise mask (same as the Flux2 inpainter's mask blur). Soft edges translate directly into per-row denoise strength on H3."),
                io.Float.Input("blend_pixels", default=16.0, min=0.0, max=256.0, step=0.5, advanced=True,
                               tooltip="Feather width of the blend mask used when stitching the decoded still back into the original image."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               advanced=True,
                               tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the render area; 'max' uses a 2048px short edge for best identity fidelity but is slower."),
                # ---------------- Connectors ----------------
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image",
                                                           tooltip="Extra reference image - becomes <Picture 2>, <Picture 3>, ..."),
                                      prefix="ref_image_", min=0, max=8)),
                io.Sigmas.Input("sigmas", optional=True,
                                tooltip="Optional exact sigma schedule (e.g. from a '⭐ Star Custom Sigmas' node) - overrides the scheduler, steps and denoise widgets."),
                io.Custom("STAR_PREVIEW").Input("preview", optional=True,
                                                tooltip="Optional live sampling preview from a '⭐ Star Preview' node."),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE",
                                tooltip="The original image with the inpainted region stitched back in."),
                io.Latent.Output(display_name="LATENT",
                                 tooltip="The sampled packed AV latent (NestedTensor video+audio), before decoding."),
                io.Mask.Output(display_name="MASK"),
                io.Int.Output(display_name="seed"),
            ],
        )

    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, model, clip, vae, image, mask, prompt, reference, megapixels,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                negative_prompt, expand_pixels, mask_blur, blend_pixels,
                ref_image_size, ref_images=None, sigmas=None, preview=None) -> io.NodeOutput:
        if image.shape[0] != 1:
            raise ValueError("[Star Minimax Inpainter] MiniMax H3 supports batch size 1 - feed one image at a time.")
        if not prompt.strip():
            prompt = DEFAULT_PROMPT

        with torch.no_grad():
            # 1. Crop image + mask to the inpaint context area (the shared
            #    Flux2/Qwen inpainter crop & stitch implementation)
            helper = StarFlux2Inpainter()
            crop_info, cropped_image, cropped_mask = helper.crop_image_and_mask(
                image, mask, context_expand_pixels=expand_pixels,
                blur_mask_pixels=mask_blur, blend_pixels=blend_pixels)
            if megapixels != "native":
                cropped_image, cropped_mask = _scale_to_area(
                    cropped_image, cropped_mask, float(megapixels))
            width, height = cropped_image.shape[2], cropped_image.shape[1]
            logging.info("[Star Minimax Inpainter] render canvas %dx%d (%.2f MP)",
                         width, height, width * height / 1048576)

            # 2. Image-mode AV latent: the source still's first video token as
            #    latent_image (preserved rows converge back to it), plus the
            #    silent audio member the packed layout always carries
            video = vae.encode(cropped_image[..., :3])[:, :, :1]
            audio_t = max(1, round(1.0 / FPS * AUDIO_LATENT_FPS))
            audio = torch.zeros([1, 32, 2, audio_t],
                                device=comfy.model_management.intermediate_device())
            latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

            # Denoise mask packed like the latent: video mask (1 = regenerate),
            # audio pinned to silence
            denoise_mask = comfy.nested_tensor.NestedTensor(
                (cropped_mask[:1], torch.zeros_like(audio)))

            # 3. ref2va conditioning: <Picture 1> is the source crop, the
            #    connected reference images follow as <Picture 2>, ...
            ref_items, ref_blocks = [], []
            if reference != "off":
                ref_src = cropped_image
                if reference == "masked source":
                    m = cropped_mask[:1].unsqueeze(-1)
                    ref_src = cropped_image[:1] * (1.0 - m) + 0.5 * m
                item, block = _image_ref(ref_src, vae, ref_image_size, width, height)
                ref_items.append(item)
                ref_blocks.append(block)
            for img in (ref_images or {}).values():
                if img is None:
                    continue
                item, block = _image_ref(img, vae, ref_image_size, width, height)
                ref_items.append(item)
                ref_blocks.append(block)
            if ref_items:
                logging.info("[Star Minimax Inpainter] %d reference image(s), <Picture 1> = %s",
                             len(ref_items), reference)

            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

            # Negative conditioning only when real CFG is requested (at 1.0 the
            # basic positive-only guider samples exactly like the AIO)
            negative = None
            if cfg != 1.0:
                neg_tokens = clip.tokenize(negative_prompt, minimax_ref_items=ref_items)
                negative = clip.encode_from_tokens_scheduled(neg_tokens)
                if ref_blocks:
                    negative = node_helpers.conditioning_set_values(negative, {"minimax_refs": ref_blocks})

            # 4. Sampling on the packed AV latent (RandomNoise + guider +
            #    KSamplerSelect + SamplerCustomAdvanced, in-process); a SIGMAS
            #    link replaces the scheduler/steps/denoise widgets
            if sigmas is not None:
                if sigmas.numel() < 2:
                    logging.warning("[Star Minimax Inpainter] custom sigmas ignored (need at least 2 sigma values)")
                    sigmas = None
                else:
                    logging.info("[Star Minimax Inpainter] custom sigmas, %d steps", sigmas.numel() - 1)
            if sigmas is None:
                sigmas = StarMinimaxAllInOne._get_sigmas(model, scheduler, steps, denoise)
            sample_model = apply_star_preview(model, preview) if preview is not None else model
            samples = run_sample(sample_model, cond, latent, seed, sampler_name, sigmas,
                                 negative=negative, cfg=cfg, denoise_mask=denoise_mask)

            # 5. Still decode + stitch back into the original image
            decoded = decode_still(vae, samples)
            blend = crop_info["cropped_mask_blend"]
            if (decoded.shape[2], decoded.shape[1]) != (blend.shape[-1], blend.shape[-2]):
                decoded = _resize(decoded, blend.shape[-1], blend.shape[-2], "disabled")
            stitched = helper.stitch_image(crop_info, decoded)

        torch.cuda.empty_cache()
        return io.NodeOutput(stitched, {"samples": samples}, mask, seed)


NODE_CLASS_MAPPINGS = {
    "StarMinimaxH3ImageInpainter": StarMinimaxH3ImageInpainter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxH3ImageInpainter": "⭐ Star Minimax H3 Image Inpainter",
}
