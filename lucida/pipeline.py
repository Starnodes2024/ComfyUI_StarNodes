# --------------------------------------------------------------------------------
# Portions of this file incorporate code from the Lucida project
# (https://github.com/egeorcun/lucida), which is licensed under the MIT License:
#
# Copyright (c) 2024 Ege Orcun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------------------------------------

"""Lucida inference pipeline — ComfyUI-agnostic core.

Faithful port of the official pipeline from https://github.com/egeorcun/lucida
(`bgr/segmenter.py`, `bgr/refiner.py`, `bgr/decontaminate.py`, `bgr/pipeline.py`),
adapted to load weights straight from a local file so no Hugging Face download
is ever required.

Model contract: predict_alpha(PIL.Image) -> np.float32 (H, W) in [0, 1] at the
input image's own resolution. Input resolution 1024x1024 (as trained), ImageNet
normalization, last head + sigmoid — identical to the upstream implementation.
"""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 1024

_CHECKPOINT_WRAPPER_KEYS = ("model", "state_dict", "weights")


def _load_state_dict_file(path: str) -> dict:
    """Load a state dict from .safetensors or a torch checkpoint.

    Handles the upstream training-checkpoint format
    ``torch.save({"model": state_dict, "optimizer": ..., "epoch": ...})``
    as well as plain state dicts, and strips the ``_orig_mod.`` (torch.compile)
    and ``module.`` (DDP) prefixes exactly like upstream does.
    """
    lower = path.lower()
    if lower.endswith(".safetensors") or lower.endswith(".sft"):
        from safetensors.torch import load_file

        state_dict = load_file(path, device="cpu")
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Training checkpoints may contain optimizer/scheduler objects.
            payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload
        if isinstance(payload, dict):
            for key in _CHECKPOINT_WRAPPER_KEYS:
                if key in payload and isinstance(payload[key], dict):
                    state_dict = payload[key]
                    break
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


class LucidaModel:
    """BiRefNet (lucida fine-tune) loaded from a local weights file."""

    def __init__(self, weights_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
        from .birefnet import BiRefNet
        from .BiRefNet_config import BiRefNetConfig

        self.device = device
        self.dtype = dtype if device.type != "cpu" else torch.float32
        self.input_size = INPUT_SIZE

        model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
        state_dict = _load_state_dict_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        """PIL RGB image -> float32 alpha (H, W) in [0, 1] at input resolution."""
        rgb = image.convert("RGB")
        inp = self.transform(rgb).unsqueeze(0).to(self.device, self.dtype)
        preds = self.model(inp)[-1].float().sigmoid().cpu()
        alpha = transforms.functional.resize(preds[0], rgb.size[::-1])[0]
        return alpha.clamp(0, 1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Edge refinement — direct port of bgr/refiner.py (CGM-style): crops the
# regions the model is unsure about, re-asks the same model at higher
# effective resolution, and feather-blends the result only inside the
# uncertain band.
# ---------------------------------------------------------------------------


def _regions(band: np.ndarray, min_region: int, max_patches: int):
    labels, num = ndimage.label(ndimage.binary_dilation(band, iterations=4))
    if num == 0:
        return []
    sizes = ndimage.sum(band, labels, range(1, num + 1))
    order = np.argsort(sizes)[::-1]
    boxes = ndimage.find_objects(labels)
    out = []
    for i in order[:max_patches]:
        if sizes[i] < min_region:
            break
        sl = boxes[i]
        out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


def refine_alpha(
    segmenter: LucidaModel,
    image: Image.Image,
    alpha: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    min_region: int = 256,
    context: float = 0.35,
    max_patches: int = 6,
) -> np.ndarray:
    if (image.size[1], image.size[0]) != alpha.shape:
        raise ValueError(
            f"image size {image.size[::-1]} does not match alpha shape {alpha.shape}"
        )
    h, w = alpha.shape
    band = (alpha > low) & (alpha < high)
    out = alpha.copy()
    rgb = image.convert("RGB")
    for y0, y1, x0, x1 in _regions(band, min_region, max_patches):
        cy, cx = int((y1 - y0) * context), int((x1 - x0) * context)
        yy0, yy1 = max(0, y0 - cy), min(h, y1 + cy)
        xx0, xx1 = max(0, x0 - cx), min(w, x1 + cx)
        crop = rgb.crop((xx0, yy0, xx1, yy1))
        refined = segmenter.predict_alpha(crop)
        # feather: soften the band mask, blend only inside the band
        local_band = band[yy0:yy1, xx0:xx1].astype(np.float32)
        weight = ndimage.gaussian_filter(local_band, 2).clip(0, 1)
        weight[local_band == 0] = 0.0
        region = out[yy0:yy1, xx0:xx1]
        out[yy0:yy1, xx0:xx1] = weight * refined + (1 - weight) * region
    return out.clip(0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Color decontamination. Official path uses pymatting's multi-level foreground
# estimation (bgr/decontaminate.py). If pymatting is not installed we fall
# back to a nearest-solid-color bleed (the classic "defringe" behaviour), so
# the node keeps working; install pymatting for the exact upstream result.
# ---------------------------------------------------------------------------

_fallback_notice_shown = False


def _decontaminate_fallback(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Edge color-spill cleanup without pymatting.

    Pixels that are not fully opaque inherit the colour of the nearest fully
    opaque pixel (scipy distance transform), which removes the background
    colour halo along semi-transparent edges. Alpha is left untouched.
    """
    global _fallback_notice_shown
    if not _fallback_notice_shown:
        print(
            "[ComfyUI-Lucida] pymatting not installed — using the built-in "
            "nearest-colour decontamination fallback. For the official "
            "decontamination quality: pip install pymatting"
        )
        _fallback_notice_shown = True
    solid = alpha >= 0.999
    if not solid.any():
        return rgb
    indices = ndimage.distance_transform_edt(~solid, return_indices=True)
    nearest_fg = rgb[indices[0], indices[1]]
    contaminated = (alpha < 0.999)[..., None]
    return np.where(contaminated, nearest_fg, rgb)


def decontaminate(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Return an RGBA PIL image whose RGB is cleaned of background colour spill."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if alpha.shape != rgb.shape[:2]:
        raise ValueError(f"alpha shape {alpha.shape} != image {rgb.shape[:2]}")
    try:
        from pymatting import estimate_foreground_ml
    except ImportError:
        fg = _decontaminate_fallback(rgb.astype(np.float32), alpha)
    else:
        fg = np.clip(estimate_foreground_ml(rgb, alpha.astype(np.float64)), 0, 1)
    out = np.dstack([np.clip(fg, 0, 1), alpha.clip(0, 1)])
    return Image.fromarray(np.round(out * 255).astype(np.uint8), mode="RGBA")


def compose_rgba(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """RGBA without decontamination: original RGB + predicted alpha."""
    rgba = image.convert("RGB").copy()
    rgba.putalpha(Image.fromarray(np.round(alpha * 255).astype(np.uint8)))
    return rgba
