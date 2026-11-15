"""Star Face Detailer - local YOLO face refinement without external nodes.

The same engine powers this standalone node and the optional per-box face
refinement in StarRegionalSampler. Only the selected face area is pasted back;
the surrounding image is left untouched.
"""

import logging
import os
import shutil
import subprocess
import sys
import urllib.request

import torch
import torch.nn.functional as F

import folder_paths
import node_helpers
import comfy.hooks
import comfy.model_management
import comfy.samplers
import comfy.text_encoders.fibo2
import comfy.utils
from comfy_extras.nodes_moss_image import TextEncodeMOSSImageEdit
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from nodes import CLIPTextEncode, common_ksampler

from .star_regional_spatial_lora_canvas import _region_hooks

DEFAULT_DETECTOR = "bbox/face_yolov8s.pt"
DEFAULT_DETECTOR_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt"
_DETECTOR_CACHE = {}


def _model_roots():
    roots = set()
    for kind in ("checkpoints", "diffusion_models"):
        for path in folder_paths.get_folder_paths(kind):
            roots.add(os.path.dirname(path))
    return roots


def _detector_files():
    found = {}
    for root in _model_roots():
        for folder in ("ultralytics", "detection"):
            base = os.path.join(root, folder)
            if not os.path.isdir(base):
                continue
            for current, _dirs, files in os.walk(base):
                for name in files:
                    if not name.lower().endswith(".pt") or "face" not in name.lower():
                        continue
                    path = os.path.join(current, name)
                    label = os.path.relpath(path, base).replace(os.sep, "/")
                    found[label] = path
    if DEFAULT_DETECTOR not in found:
        for root in _model_roots():
            path = os.path.join(root, "ultralytics", DEFAULT_DETECTOR.replace("/", os.sep))
            if os.path.isfile(path):
                found[DEFAULT_DETECTOR] = path
                break
    return found


def _detector_choices():
    files = _detector_files()
    return sorted(files, key=lambda name: (name != DEFAULT_DETECTOR, name.lower())) or [DEFAULT_DETECTOR]


def _download_default_detector():
    existing = _detector_files().get(DEFAULT_DETECTOR)
    if existing is not None:
        return existing
    root = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
    path = os.path.join(root, "ultralytics", DEFAULT_DETECTOR.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".download"
    logging.warning(f"[Star Face Detailer] {DEFAULT_DETECTOR} not found - downloading 22.5 MB from Bingsu/adetailer to {path}")
    try:
        with urllib.request.urlopen(DEFAULT_DETECTOR_URL, timeout=30) as response, open(temporary, "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if os.path.getsize(temporary) < 1024 * 1024:
            raise RuntimeError("the downloaded detector is unexpectedly small")
        os.replace(temporary, path)
    except Exception as e:
        if os.path.isfile(temporary):
            os.remove(temporary)
        raise RuntimeError(
            f"Could not download {DEFAULT_DETECTOR} ({e}). Download it manually from "
            f"{DEFAULT_DETECTOR_URL} and place it in models/ultralytics/bbox/.") from e
    logging.info(f"[Star Face Detailer] default detector downloaded to {path}")
    return path


def _yolo_class():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        logging.warning("[Star Face Detailer] ultralytics is missing - installing it now")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0,<9.0.0"])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Could not install ultralytics automatically ({e}). Run: {sys.executable} -m pip install 'ultralytics>=8.3.0,<9.0.0'") from e
        from ultralytics import YOLO
        return YOLO


def _detect(image, detector_model, threshold):
    path = _detector_files().get(detector_model)
    if path is None and detector_model == DEFAULT_DETECTOR:
        path = _download_default_detector()
    if path is None:
        available = ", ".join(_detector_choices())
        raise ValueError(f"Face detector '{detector_model}' was not found. Available: {available}")
    if path not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[path] = _yolo_class()(path)
    detector = _DETECTOR_CACHE[path]
    pixels = (image[0, ..., :3].detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    device = comfy.model_management.get_torch_device()
    result = detector.predict(source=pixels, conf=threshold, device=device.index if device.type == "cuda" else "cpu", verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    return [(list(map(float, box)), float(score)) for box, score in zip(boxes, scores)]


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


def _resize_image(image, width, height):
    return comfy.utils.common_upscale(image.detach().permute(0, 3, 1, 2), width, height, "lanczos", "disabled").permute(0, 2, 3, 1)


def _reference_conditioning(model, clip, vae, prompt, negative_prompt, reference_image, guide_size):
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
                sampler_name, scheduler, denoise, guide_size, crop_factor, feather):
    _batch, height, width, _channels = image.shape
    left, top, right, bottom = _crop_box(box, (width, height), crop_factor)
    crop = image[:, top:bottom, left:right]
    guide = _resize_image(crop, guide_size, guide_size)
    sampled = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                              positive, negative, {"samples": vae.encode(guide)}, denoise=denoise)[0]
    refined = _resize_image(vae.decode(sampled["samples"]), right - left, bottom - top)
    refined = refined[..., :crop.shape[-1]]  # Qwen 2.1 decodes RGBA; keep the input's channel count
    mask = _paste_mask((width, height), box, feather)[:, :, top:bottom, left:right]
    mask = mask.permute(0, 2, 3, 1).to(device=image.device, dtype=image.dtype)
    image[:, top:bottom, left:right] = crop * (1.0 - mask) + refined * mask
    return image


def detail_image(image, model, vae, positive, negative, seed, steps, cfg,
                 sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR,
                 guide_size=512, crop_factor=3.0, feather=16, threshold=0.5):
    image = image.clone()
    detailed = 0
    for batch in range(image.shape[0]):
        detections = _detect(image[batch:batch + 1], detector_model, threshold)
        detections.sort(key=lambda item: (item[0][0], item[0][1]))
        for index, (box, _score) in enumerate(detections):
            image[batch:batch + 1] = _detail_one(
                image[batch:batch + 1], box, model, vae, positive, negative,
                seed + detailed + index, steps, cfg, sampler_name, scheduler,
                denoise, guide_size, crop_factor, feather)
            detailed += 1
    logging.info(f"[Star Face Detailer] {detailed} face(s) refined" if detailed else "[Star Face Detailer] no faces detected - image unchanged")
    return image


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
        for detection_index, (box, score) in enumerate(detections):
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
        assignments.append((region_index, detections[detection_index][0]))
    return assignments


def detail_regions(image, model, clip, vae, regions, negative_prompt, seed, steps, cfg,
                   sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR):
    image = image.clone()
    encoder = CLIPTextEncode()
    detailed = 0
    for batch in range(image.shape[0]):
        _batch, height, width, _channels = image.shape
        detections = _detect(image[batch:batch + 1], detector_model, 0.5)
        assignments = {index: box for index, box in _region_assignments(regions, detections, (width, height))}
        for index, region in enumerate(regions):
            box = assignments.get(index)
            if box is None:
                logging.info(f"[Star Regional Sampler] face detail box {index + 1}: no matching face detected - skipped")
                continue
            prompt = region.get("prompt", "").strip()
            detail_prompt = f"{prompt}, high quality face details" if prompt else "high quality face details"
            positive = encoder.encode(clip, detail_prompt)[0]
            negative = encoder.encode(clip, negative_prompt)[0]
            token = region.get("token")
            if token:
                positive = comfy.hooks.set_hooks_for_conditioning(positive, _region_hooks(token), append_hooks=False)
                negative = comfy.hooks.set_hooks_for_conditioning(negative, _region_hooks(token), append_hooks=False)
            image[batch:batch + 1] = _detail_one(
                image[batch:batch + 1], box, model, vae, positive, negative,
                seed + index, steps, cfg, sampler_name, scheduler, denoise, 512, 3.0, 16)
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
                "prompt": ("STRING", {"default": "high quality face details", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "detector_model": (_detector_choices(), {"default": DEFAULT_DETECTOR, "tooltip": "Local YOLO face detector from models/ultralytics or models/detection. No provider nodes are needed."}),
                "guide_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.2, "max": 4.0, "step": 0.1}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
                "reference_image": ("IMAGE", {"tooltip": "Optional identity reference for Fibo2, MOSS Image Edit or Qwen Image 2.1. Use a swap prompt and denoise 0.75-1.0 for face replacement."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def detail(self, image, model, clip, vae, prompt, negative_prompt, seed, steps, cfg,
               sampler_name, scheduler, denoise, detector_model=DEFAULT_DETECTOR,
               guide_size=512, crop_factor=3.0, feather=16, threshold=0.5, reference_image=None):
        if reference_image is not None:
            positive, negative = _reference_conditioning(model, clip, vae, prompt, negative_prompt, reference_image, guide_size)
        else:
            encoder = CLIPTextEncode()
            positive = encoder.encode(clip, prompt)[0]
            negative = encoder.encode(clip, negative_prompt)[0]
        return (detail_image(image, model, vae, positive, negative, seed, steps, cfg,
                             sampler_name, scheduler, denoise, detector_model,
                             guide_size, crop_factor, feather, threshold),)


NODE_CLASS_MAPPINGS = {"StarFaceDetailer": StarFaceDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {"StarFaceDetailer": "⭐ Star Face Detailer"}
