"""Star Face Detailer - WeVisOne-VL guided refinement without external nodes.

The same engine powers this standalone node and the optional per-box face
refinement in StarRegionalSampler. Only the selected face area is pasted back;
the surrounding image is left untouched.
"""

import importlib.util
import logging
import os
import shutil
import sys
import urllib.request

import torch
import torch.nn.functional as F

import folder_paths
import node_helpers
import comfy.hooks
import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.text_encoders.fibo2
import comfy.utils
from comfy_extras.nodes_moss_image import TextEncodeMOSSImageEdit
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from comfy_extras.nodes_wevisone import WeVisOneDetect
from nodes import CLIPTextEncode, common_ksampler

from .star_regional_spatial_lora_canvas import _region_hooks


def _find_apply_star_preview():
    # ComfyUI registers node packs in sys.modules under their folder path, not
    # the package name, so "comfyui_starnodes.misc.star_preview" cannot be
    # imported directly. The pack imports it on load through star_sampler, so
    # find the already-loaded module.
    for name, module in sys.modules.items():
        if "comfyui_starnodes" in name and name.endswith("misc.star_preview"):
            return getattr(module, "apply_star_preview", None)
    # Pack present but the module was never imported - load it by file path.
    preview_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "comfyui_starnodes", "misc", "star_preview.py")
    if os.path.isfile(preview_file):
        spec = importlib.util.spec_from_file_location("starnodes_star_preview", preview_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "apply_star_preview", None)
    return None


apply_star_preview = _find_apply_star_preview()

DETAIL_PROMPT = "high quality face details, natural skin texture, fine pores, photorealistic, preserve original skin tone and facial identity"
REFINE_TARGETS = ["face", "eyes", "mouth", "nose", "ears", "breast", "body", "custom"]
WEVISONE_CLIPS = [
    "wevisone_vl_2b_int8_convrot.safetensors",  # 2.7 GB - smallest, default
    "wevisone_vl_4b_int8_convrot.safetensors",  # 4.9 GB - better quality
    "wevisone_vl_4b_bf16.safetensors",          # 9.0 GB - best quality
]
DEFAULT_DETECTOR = WEVISONE_CLIPS[0]
WEVISONE_CLIP_URL = "https://huggingface.co/Comfy-Org/WeVisOne-VL/resolve/main/text_encoders/{}"
_WEVISONE_CACHE = {}
_REFINER_CACHE = {}
_LORA_CACHE = {}

REFINER_INSTRUCTION = (
    "You write prompts for an image-generation detailer. Look at the {target} in this image "
    "and write ONE short prompt (40 words maximum) for regenerating exactly this {target}. "
    "Combine what you see (key visual features, exact skin tone and complexion, expression "
    "where applicable, angle, lighting) with the user's instructions, which you must always "
    "respect and include in the output: \"{prompt}\". The prompt must preserve the person's "
    "identity and exact skin tone. Describe only what is visible, no names, no explanations, "
    "no markdown - output the prompt text only."
)

REFINER_INSTRUCTION_TEXT_ONLY = (
    "You refine prompts for an image-generation detailer. Rewrite the user's instructions into "
    "ONE short, concrete image prompt (40 words maximum) about the {target}, keeping every user "
    "instruction and adding a few natural detail descriptors (texture, lighting). User "
    "instructions: \"{prompt}\". No explanations, no markdown - output the prompt text only."
)


def _detector_choices():
    return list(WEVISONE_CLIPS)


def _download_model(url, path, description):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".download"
    logging.warning(f"[Star Face Detailer] {description} not found - downloading {url} to {path}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if os.path.getsize(temporary) < 1024 * 1024:
            raise RuntimeError("the download is unexpectedly small")
        os.replace(temporary, path)
    except Exception as e:
        if os.path.isfile(temporary):
            os.remove(temporary)
        raise RuntimeError(
            f"Could not download {description} ({e}). Download it manually from {url} and place it at {path}.") from e
    logging.info(f"[Star Face Detailer] downloaded {description} to {path}")
    return path


def _wevisone_clip(name=DEFAULT_DETECTOR):
    if name not in _WEVISONE_CACHE:
        path = folder_paths.get_full_path("text_encoders", name)
        if path is None:
            root = folder_paths.get_folder_paths("text_encoders")[0]
            path = _download_model(WEVISONE_CLIP_URL.format(name), os.path.join(root, name),
                                   f"WeVisOne-VL text encoder {name}")
        logging.warning(f"[Star Face Detailer] loading WeVisOne-VL encoder {name} (first use may take a moment)")
        _WEVISONE_CACHE[name] = comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    return _WEVISONE_CACHE[name]


def _refiner_files():
    return sorted(folder_paths.get_filename_list("text_encoders"))


def _load_refiner(name):
    if name not in _REFINER_CACHE:
        path = folder_paths.get_full_path_or_raise("text_encoders", name)
        logging.warning(f"[Star Face Detailer] loading prompt refiner {name} (stays loaded for reuse)")
        clip = comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
        if not hasattr(clip.cond_stage_model, "generate"):
            raise RuntimeError(
                f"{name} does not support text generation. Choose an instruct/VL encoder "
                "(Qwen3-VL, Qwen3.5, WeVisOne, Gemma, ...) as prompt refiner.")
        _REFINER_CACHE[name] = clip
    return _REFINER_CACHE[name]


def _tokens_have_image(tokens):
    for entry in tokens.values():
        for batch in entry:
            for item in batch:
                if isinstance(item[0], dict) and item[0].get("type") == "image":
                    return True
    return False


def _model_with_lora(model, lora_name, strength):
    # Clones the base model with the LoRA patched in; the LoRA weights are read
    # from disk only once. Used per face, the original model stays untouched.
    if lora_name not in _LORA_CACHE:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        _LORA_CACHE[lora_name] = comfy.utils.load_torch_file(path, safe_load=True)
    patched, _ = comfy.sd.load_lora_for_models(model, None, _LORA_CACHE[lora_name], strength, 0)
    return patched


def _refine_face_prompt(refiner, crop, user_prompt, target="face"):
    # One greedy pass turns the crop + the user's general style prompt into a
    # short prompt describing this exact target. Encoders without image support
    # refine the user's prompt text only.
    prompt = user_prompt.strip() or DETAIL_PROMPT
    # skip_template=False is required for Gemma4 (its tokenizer skips the chat
    # template by default, which makes it answer with zero tokens); Qwen applies
    # its template by default anyway and plain tokenizers ignore the kwarg.
    tokens = refiner.tokenize(REFINER_INSTRUCTION.format(prompt=prompt, target=target), image=crop, skip_template=False)
    if not _tokens_have_image(tokens):
        tokens = refiner.tokenize(REFINER_INSTRUCTION_TEXT_ONLY.format(prompt=prompt, target=target), skip_template=False)
    ids = refiner.generate(tokens, do_sample=True, max_length=96, top_k=1, repetition_penalty=1.05, seed=0)
    return " ".join(refiner.decode(ids).strip().strip('"').split())


def _detect_wevisone(image, query="face", encoder=DEFAULT_DETECTOR):
    # One VL pass segments every matching instance at once: individual_masks
    # gives one alpha matte per instance in bboxes order, so the refine loop
    # still processes each detected target separately.
    clip = _wevisone_clip(encoder)
    output = WeVisOneDetect.execute(clip, image, "segmentation", query, True)
    masks, boxes = output.result[0], output.result[1][0]
    detections = []
    for i, box in enumerate(boxes):
        seg = masks[i].detach().float().cpu() if i < len(masks) else None
        detections.append(([box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]], 1.0, seg))
    logging.info(f"[Star Face Detailer] WeVisOne-VL found {len(detections)} {query} instance(s)")
    return detections


def _detect(image, detector_model=DEFAULT_DETECTOR, query="face"):
    if detector_model not in WEVISONE_CLIPS:
        # Workflows saved before the YOLO removal hold a YOLO file or the old
        # sentinel string here - fall back to the default WeVisOne encoder.
        logging.info(f"[Star Face Detailer] detector '{detector_model}' is no longer supported - using {DEFAULT_DETECTOR}")
        detector_model = DEFAULT_DETECTOR
    return _detect_wevisone(image, query, detector_model)


def _crop_box(box, image_size, crop_factor):
    width, height = image_size
    x0, y0, x1, y1 = box
    side = min(max(x1 - x0, y1 - y0) * crop_factor, min(width, height))
    side = max(16.0, side)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    left = max(0.0, min(width - side, cx - side * 0.5))
    top = max(0.0, min(height - side, cy - side * 0.5))
    return round(left), round(top), round(left + side), round(top + side)


def _paste_mask(image_size, box, feather):
    # The detected rectangle only positions a soft head-shaped mask. Neither the
    # canvas box nor a hard detector rectangle is pasted into the final image.
    width, height = image_size
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5 - (y1 - y0) * 0.05
    radius_x = max((x1 - x0) * 0.65, 1.0)
    radius_y = max((y1 - y0) * 0.75, 1.0)
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                            torch.arange(width, dtype=torch.float32), indexing="ij")
    distance = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    mask = (distance <= 1.0).to(torch.float32).unsqueeze(0).unsqueeze(0)
    feather = round(feather)
    if feather > 0:
        mask = F.avg_pool2d(F.pad(mask, (feather,) * 4, mode="replicate"), 2 * feather + 1, stride=1)
    return mask


def _seg_paste_mask(seg, image_size, box, feather):
    """Paste mask from a real segmentation mask (float [h,w], values 0..1).
    The soft alpha-matte edge is preserved (no binarization). The matte is
    slightly ERODED, not dilated: the paste follows the face silhouette and the
    blend hides inside the original skin. Dilation filled the whole crop
    square, which produced visible square patches."""
    width, height = image_size
    if tuple(seg.shape[-2:]) != (height, width):
        seg = F.interpolate(seg[None, None], (height, width), mode="bilinear", align_corners=False)[0, 0]
    mask = seg.clamp(0.0, 1.0).to(torch.float32).unsqueeze(0).unsqueeze(0)
    shrink = max(1, round((box[2] - box[0]) * 0.03))
    mask = -F.max_pool2d(-F.pad(mask, (shrink,) * 4, mode="replicate"), 2 * shrink + 1, stride=1)
    feather = round(feather)
    if feather > 0:
        mask = F.avg_pool2d(F.pad(mask, (feather,) * 4, mode="replicate"), 2 * feather + 1, stride=1)
    return mask


def _face_paste_mask(image_size, box, feather, seg=None):
    if seg is not None:
        return _seg_paste_mask(seg, image_size, box, feather)
    return _paste_mask(image_size, box, feather)


def _edge_fade(mask, fade):
    """Multiplies a [1,1,h,w] mask by a border ramp so it is exactly 0 at the
    crop edges. Without this, any mask that reaches the crop border gets
    clipped and the refined/original pixels meet along a visible square seam."""
    h, w = mask.shape[2], mask.shape[3]
    fade = min(round(fade), w // 2, h // 2)
    if fade <= 0:
        return mask
    ramp = torch.linspace(0.0, 1.0, fade, dtype=mask.dtype)
    wx = torch.ones(w, dtype=mask.dtype)
    wy = torch.ones(h, dtype=mask.dtype)
    wx[:fade] = ramp
    wx[w - fade:] = ramp.flip(0)
    wy[:fade] = ramp
    wy[h - fade:] = ramp.flip(0)
    return mask * (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0).to(mask.device, mask.dtype)


def _feather_for_crop(feather, crop_side):
    # Feather scales up with the crop for large images, but small targets
    # (eyes, mouth) need a proportionally small blend zone too.
    return min(max(round(feather), round(crop_side / 16)), max(2, round(crop_side / 8)))


def _color_match(refined, original, mask, strength):
    # Per-channel mean/std transfer inside the paste mask: anchors the refined
    # region to the original skin tone/lighting instead of letting the model
    # drift to its own complexion prior. strength 0 disables the correction.
    m = (mask > 0.5).to(refined.dtype)
    count = m.sum().clamp(min=1.0)
    if count <= 1.0:
        return refined
    mean_r = (refined * m).sum(dim=(1, 2), keepdim=True) / count
    mean_o = (original * m).sum(dim=(1, 2), keepdim=True) / count
    std_r = (((refined - mean_r) ** 2 * m).sum(dim=(1, 2), keepdim=True) / count).sqrt()
    std_o = (((original - mean_o) ** 2 * m).sum(dim=(1, 2), keepdim=True) / count).sqrt()
    # Clamp the contrast ratio: low-contrast refined crops (flat backgrounds)
    # would otherwise amplify deviations far beyond a natural correction.
    scale = ((std_o + 1e-5) / (std_r + 1e-5)).clamp(0.5, 2.0)
    matched = (refined - mean_r) * scale + mean_o
    return (refined * (1.0 - strength) + matched * strength).clamp(0.0, 1.0)


def _resize_image(image, width, height):
    return comfy.utils.common_upscale(image.detach().permute(0, 3, 1, 2), width, height, "lanczos", "disabled").permute(0, 2, 3, 1)


def _reference_conditioning(model, clip, vae, prompt, negative_prompt, reference_image, guide_size):
    if guide_size <= 0:
        guide_size = 512
    reference = comfy.utils.common_upscale(
        reference_image[:1].detach().movedim(-1, 1), guide_size, guide_size, "lanczos", "center").movedim(1, -1)
    model_type = model.model.__class__.__name__
    if model_type == "Fibo2":
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=[reference]))
        negative_text = comfy.text_encoders.fibo2.TEMPLATE.format(negative_prompt) if negative_prompt.strip() else ""
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
        reference_latent = vae.encode(reference)
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [reference_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [reference_latent]}, append=True)
    elif model_type == "MOSSImage":
        images = {"image_1": reference}
        positive = TextEncodeMOSSImageEdit.execute(clip, prompt, "ai-toolkit", images, vae=vae)[0]
        negative = TextEncodeMOSSImageEdit.execute(clip, negative_prompt, "ai-toolkit", images, vae=vae)[0]
    elif model_type == "QwenImage21":
        positive = TextEncodeQwenImage21.execute(clip, prompt, negative_prompt, vae=vae, resolution=guide_size, images={"image_1": reference})[0]
        negative = TextEncodeQwenImage21.execute(clip, negative_prompt, "", vae=vae, resolution=guide_size, images={"image_1": reference})[0]
    else:
        raise ValueError(f"reference_image needs a Fibo2, MOSS Image Edit or Qwen Image 2.1 compatible model, not {model_type}")
    logging.info(f"[Star Face Detailer] reference identity conditioning enabled ({model_type})")
    return positive, negative


def _detail_one(image, box, model, vae, positive, negative, seed, steps, cfg,
                sampler_name, scheduler, denoise, guide_size, crop_factor, feather, seg=None,
                color_match=1.0, whole_color=False):
    _batch, height, width, _channels = image.shape
    left, top, right, bottom = _crop_box(box, (width, height), crop_factor)
    crop_side = right - left
    # guide_size 0 = auto: use the real crop size so 2-3MP renders are not
    # downsampled to 512 first (that mismatch caused seams).
    if guide_size <= 0:
        guide_size = max(512, min(1024, round(crop_side / 32) * 32))
    # Feather scales with the crop: the old fixed 16 px is invisible on
    # multi-MP images, which is what produced the visible seam ring.
    feather = _feather_for_crop(feather, crop_side)
    crop = image[:, top:bottom, left:right]
    guide = _resize_image(crop, guide_size, guide_size)
    sampled = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                              positive, negative, {"samples": vae.encode(guide)}, denoise=denoise)[0]
    refined = vae.decode(sampled["samples"])
    if refined.dim() == 5:
        # Video VAEs (WanVAE on Krea2) return [B, T, H, W, C] - take the frame.
        refined = refined[:, 0]
    refined = _resize_image(refined, right - left, bottom - top)
    refined = refined[..., :crop.shape[-1]]  # Qwen 2.1 decodes RGBA; keep the input's channel count
    mask = _face_paste_mask((width, height), box, feather, seg)[:, :, top:bottom, left:right]
    mask = _edge_fade(mask, max(8, round(crop_side / 12)))
    mask = mask.permute(0, 2, 3, 1).to(device=image.device, dtype=image.dtype)
    if color_match > 0:
        # Reference swaps use whole-crop statistics (face + hair + background),
        # which harmonizes the scene's lighting without recoloring the new
        # identity into the original person's skin tone.
        stats_mask = torch.ones_like(mask) if whole_color else mask
        refined = _color_match(refined, crop, stats_mask, color_match)
    image[:, top:bottom, left:right] = crop * (1.0 - mask) + refined * mask
    return image


def detail_image(image, model, vae, positive, negative, seed, steps, cfg,
                 sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR,
                 guide_size=512, crop_factor=3.0, feather=16,
                 clip=None, prompt="", prompt_refiner=None, target="face", face_prompts=None,
                 color_match=1.0, reference_mode=False):
    image = image.clone()
    _b, height, width, _c = image.shape
    # Union of the soft ellipse masks of all detected faces (matches what is
    # actually pasted back), per batch item, ComfyUI MASK layout [B,H,W].
    mask = torch.zeros((image.shape[0], height, width), dtype=torch.float32)
    detailed = 0
    for batch in range(image.shape[0]):
        detections = _detect(image[batch:batch + 1], detector_model, query=target)
        detections.sort(key=lambda item: (item[0][0], item[0][1]))
        for index, (box, _score, seg) in enumerate(detections):
            if face_prompts is not None and index in face_prompts["skip"]:
                print(f"[Star Face Detailer] {target} {index + 1} skipped by user", flush=True)
                continue
            left, top, right, bottom = _crop_box(box, (width, height), crop_factor)
            crop_side = right - left
            face_mask = _face_paste_mask(
                (width, height), box, _feather_for_crop(feather, crop_side), seg
            )[:, :, top:bottom, left:right]
            face_mask = _edge_fade(face_mask, max(8, round(crop_side / 12)))[0, 0]
            mask[batch, top:bottom, left:right] = torch.maximum(
                mask[batch, top:bottom, left:right], face_mask.to(mask.device))
            custom = face_prompts["prompts"][index].strip() if face_prompts is not None and index < len(face_prompts["prompts"]) else ""
            base_prompt = custom or prompt
            face_model = model
            if face_prompts is not None and index < len(face_prompts["loras"]):
                lora = face_prompts["loras"][index]
                if lora is not None:
                    face_model = _model_with_lora(model, *lora)
                    print(f"[Star Face Detailer] {target} {index + 1} LoRA: {lora[0]} (strength {lora[1]:g})", flush=True)
            face_positive = positive
            if prompt_refiner is not None:
                face_prompt = _refine_face_prompt(prompt_refiner, image[batch:batch + 1, top:bottom, left:right], base_prompt, target)
                print(f"[Star Face Detailer] {target} {index + 1} refined prompt: {face_prompt}", flush=True)
                face_positive = CLIPTextEncode().encode(clip, face_prompt)[0]
            elif custom:
                print(f"[Star Face Detailer] {target} {index + 1} custom prompt: {custom}", flush=True)
                face_positive = CLIPTextEncode().encode(clip, custom)[0]
            image[batch:batch + 1] = _detail_one(
                image[batch:batch + 1], box, face_model, vae, face_positive, negative,
                seed + detailed + index, steps, cfg, sampler_name, scheduler,
                denoise, guide_size, crop_factor, feather, seg=seg, color_match=color_match,
                whole_color=reference_mode)
            detailed += 1
    logging.info(f"[Star Face Detailer] {detailed} {target} region(s) refined" if detailed else f"[Star Face Detailer] no {target} detected - image unchanged")
    return image, mask


def _region_assignments(regions, detections, image_size):
    """Assigns detected faces to canvas boxes, one-to-one, nearest center first."""
    width, height = image_size
    candidates = []
    for region_index, region in enumerate(regions):
        left = max(0.0, min(width - 1.0, region["x"] * width))
        top = max(0.0, min(height - 1.0, region["y"] * height))
        right = max(left + 1.0, min(float(width), (region["x"] + region["w"]) * width))
        bottom = max(top + 1.0, min(float(height), (region["y"] + region["h"]) * height))
        center_x, center_y = (left + right) * 0.5, (top + bottom) * 0.5
        span = max(right - left, bottom - top, 1.0)
        for detection_index, (box, score, _seg) in enumerate(detections):
            face_x, face_y = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
            inside = left <= face_x <= right and top <= face_y <= bottom
            distance = ((face_x - center_x) ** 2 + (face_y - center_y) ** 2) ** 0.5
            if not inside and distance > span * 1.5:
                continue
            candidates.append(((0 if inside else 1, distance / span, -score), region_index, detection_index))
    assignments = []
    used_regions, used_faces = set(), set()
    for _cost, region_index, detection_index in sorted(candidates):
        if region_index in used_regions or detection_index in used_faces:
            continue
        used_regions.add(region_index)
        used_faces.add(detection_index)
        assignments.append((region_index, detections[detection_index][0], detections[detection_index][2]))
    return assignments


def detail_regions(image, model, clip, vae, regions, negative_prompt, seed, steps, cfg,
                   sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR):
    image = image.clone()
    encoder = CLIPTextEncode()
    detailed = 0
    for batch in range(image.shape[0]):
        _batch, height, width, _channels = image.shape
        detections = _detect(image[batch:batch + 1], detector_model)
        assignments = {index: (box, seg) for index, box, seg in _region_assignments(regions, detections, (width, height))}
        for index, region in enumerate(regions):
            assignment = assignments.get(index)
            if assignment is None:
                logging.info(f"[Star Regional Sampler] face detail box {index + 1}: no matching face detected - skipped")
                continue
            prompt = region.get("prompt", "").strip()
            detail_prompt = f"{prompt}, {DETAIL_PROMPT}" if prompt else DETAIL_PROMPT
            positive = encoder.encode(clip, detail_prompt)[0]
            negative = encoder.encode(clip, negative_prompt)[0]
            token = region.get("token")
            if token:
                positive = comfy.hooks.set_hooks_for_conditioning(positive, _region_hooks(token), append_hooks=False)
                negative = comfy.hooks.set_hooks_for_conditioning(negative, _region_hooks(token), append_hooks=False)
            box, seg = assignment
            image[batch:batch + 1] = _detail_one(
                image[batch:batch + 1], box, model, vae, positive, negative,
                seed + index, steps, cfg, sampler_name, scheduler, denoise, 1024, 1.0, 0, seg=seg)
            detailed += 1
    logging.info(f"[Star Regional Sampler] face detailer refined {detailed} box(es)")
    return image, detailed


class StarFaceDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": DETAIL_PROMPT, "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "refine_target": (REFINE_TARGETS, {"default": "face", "tooltip": "What to detect and refine. WeVisOne-VL understands arbitrary targets (faces, eyes, mouth, body, ...)."}),
                "custom_target": ("STRING", {"default": "", "tooltip": "Used when refine_target = custom: a short noun phrase for WeVisOne-VL, e.g. 'hands', 'lips', 'tattoo', 'left eye'."}),
                "detector_model": (_detector_choices(), {"default": DEFAULT_DETECTOR, "tooltip": "WeVisOne-VL encoder from models/text_encoders, used for detection and per-instance alpha mattes. 2B int8 (2.7 GB, default) is fastest; 4B int8 (4.9 GB) better; 4B bf16 (9.0 GB) best quality. A missing encoder is downloaded automatically from Comfy-Org/WeVisOne-VL on first use."}),
                "guide_size": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 64, "tooltip": "Fixed refine resolution: every face crop is resized to this square size (small crops are upscaled), refined, then scaled back and pasted. 0 = auto (real crop size, 512-1024)."}),
                "crop_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1, "tooltip": "Size of the refined square = crop_factor x the detected face box. 1.0 = exactly the face box; higher = more surrounding context. Below 1.0 only the inner face area is refined - the blend transition stays inside the face, which can hide stubborn seams."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "Minimum feather of the paste mask in pixels. Scales up automatically with crop size (crop/16) to prevent seams on large images."}),
                "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Per-channel mean/std color transfer before blending, anchoring the refined region to the original tonality. Normal refinement: statistics inside the face mask (keeps the original complexion). With reference_image connected: statistics over the whole crop including hair/background, harmonizing the swap's lighting without recoloring the new identity. 1.0 = full correction, 0.5 = half, 0 = off."}),
                "prompt_refiner": (["none"] + _refiner_files(), {"default": "none", "tooltip": "Optional per-face prompt refinement with any text encoder from models/text_encoders. VL encoders (Qwen3-VL, Qwen3.5, WeVisOne, Gemma...) look at each face crop and write a short prompt for that exact face combined with your prompt text; encoders without image support refine your prompt text only. The refined prompt overrides the positive conditioning per face and is printed to the console. The encoder stays loaded after first use. Ignored when reference_image is connected."}),
                "face_prompts": ("STAR_FACE_PROMPTS", {"tooltip": "Optional ⭐ Star Multi Face Details: custom prompts for up to 5 faces/objects plus per-face skipping, matched in detection order (left to right, top to bottom). A slot's prompt overrides the general prompt for that face - and is also used as the user instructions for the prompt refiner when that is enabled. Empty slots fall back to the general prompt; skipped faces are left completely unrefined (and excluded from the mask output)."}),
                "reference_image": ("IMAGE", {"tooltip": "Optional identity reference for Fibo2, MOSS Image Edit or Qwen Image 2.1. Use a swap prompt and denoise 0.75-1.0 for face replacement."}),
                "positive": ("CONDITIONING", {"tooltip": "Optional: reuse the generation's positive conditioning (e.g. from your main prompt encode) so faces keep the subject/style context. Overrides the prompt text."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional: reuse the generation's negative conditioning. Overrides the negative_prompt text."}),
                "preview": ("STAR_PREVIEW", {"tooltip": "Optional ⭐ Star Preview options - shows a live sampling preview on the connected ⭐ Star Preview node (comfyui_starnodes pack)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "detail"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def detail(self, image, model, clip, vae, prompt, negative_prompt, seed, steps, cfg,
               sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR,
               guide_size=512, crop_factor=3.0, feather=16, reference_image=None,
               positive=None, negative=None, preview=None, prompt_refiner="none",
               refine_target="face", custom_target="", face_prompts=None, color_match=1.0):
        target = custom_target.strip() if refine_target == "custom" else refine_target
        if not target:
            target = "face"
        target = {"breast": "breasts"}.get(target, target)
        if prompt.strip() == DETAIL_PROMPT and target != "face":
            prompt = DETAIL_PROMPT.replace("face details", f"{target} details")
        if preview is not None:
            if apply_star_preview is None:
                logging.warning("[Star Face Detailer] preview connected but the comfyui_starnodes pack is not available - sampling without preview")
            else:
                model = apply_star_preview(model, preview)
        if reference_image is not None:
            positive, negative = _reference_conditioning(model, clip, vae, prompt, negative_prompt, reference_image, guide_size)
            if color_match > 0:
                print("[Star Face Detailer] color_match uses whole-crop statistics for reference face swap", flush=True)
            if denoise < 0.7:
                print(f"[Star Face Detailer] note: reference face swap usually needs denoise 0.75-1.0 (currently {denoise:g}) - low values keep the original identity", flush=True)
        else:
            encoder = CLIPTextEncode()
            if positive is None:
                positive = encoder.encode(clip, prompt)[0]
            if negative is None:
                negative = encoder.encode(clip, negative_prompt)[0]
        refiner = None
        if prompt_refiner != "none" and reference_image is None:
            refiner = _load_refiner(prompt_refiner)
            print(f"[Star Face Detailer] base prompt for per-face refinement: {prompt}", flush=True)
        elif reference_image is None:
            if positive is None:
                print(f"[Star Face Detailer] prompt: {prompt}", flush=True)
            else:
                print("[Star Face Detailer] using connected positive conditioning - the prompt text widget is ignored", flush=True)
        return detail_image(image, model, vae, positive, negative, seed, steps, cfg,
                            sampler_name, scheduler, denoise, detector_model,
                            guide_size, crop_factor, feather,
                            clip=clip, prompt=prompt, prompt_refiner=refiner, target=target,
                            face_prompts=face_prompts, color_match=color_match,
                            reference_mode=reference_image is not None)


class StarMultiFaceDetails:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["none"] + folder_paths.get_filename_list("loras")
        optional = {"lora_search": ("STRING", {"default": "",
                    "tooltip": "Type to filter the LoRA dropdowns below (case-insensitive substring). Clear to show all LoRAs again."})}
        for i in range(1, 6):
            optional[f"prompt_{i}"] = ("STRING", {"default": "", "multiline": True,
                "tooltip": f"Custom prompt for face/object {i} in detection order (left to right, top to bottom). Empty = use the detailer's general prompt."})
            optional[f"lora_{i}"] = (loras, {"default": "none",
                "tooltip": f"Optional LoRA applied to the diffusion model only for face/object {i}'s refine pass. Use lora_search above to filter this list."})
            optional[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"Strength of lora_{i}. 0 disables the LoRA."})
        optional["skip_faces"] = ("STRING", {"default": "",
                                  "tooltip": "Face/object numbers to leave completely unrefined, e.g. '1, 3' - in detection order. Refine only one face in a group shot by skipping all others."})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STAR_FACE_PROMPTS",)
    RETURN_NAMES = ("face_prompts",)
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = "Custom prompts and optional per-face LoRAs for up to 5 faces/objects for the ⭐ Star Face Detailer, with per-face skipping. Each prompt can also drive the prompt refiner individually."

    def build(self, skip_faces="", lora_search="", **kwargs):
        skip = set()
        for token in skip_faces.replace(",", " ").split():
            if token.isdigit():
                skip.add(int(token) - 1)
        loras = []
        for i in range(1, 6):
            name = kwargs.get(f"lora_{i}", "none")
            strength = kwargs.get(f"strength_{i}", 1.0)
            loras.append(None if name in ("", "none") or strength == 0 else (name, strength))
        return ({"prompts": [kwargs.get(f"prompt_{i}", "").strip() for i in range(1, 6)], "skip": skip, "loras": loras},)


NODE_CLASS_MAPPINGS = {"StarFaceDetailer": StarFaceDetailer, "StarMultiFaceDetails": StarMultiFaceDetails}
NODE_DISPLAY_NAME_MAPPINGS = {"StarFaceDetailer": "⭐ Star Face Detailer", "StarMultiFaceDetails": "⭐ Star Multi Face Details"}
