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

"""Star Lucida RMBG - Background removal with Lucida BiRefNet fine-tune.

Based on ComfyUI-Lucida by egeorcun
Original source: https://github.com/egeorcun/lucida
Model: https://huggingface.co/egeorcun/lucida
"""

import gc
import os

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy import model_management

from ..lucida.pipeline import LucidaModel, compose_rgba, refine_alpha
from ..lucida.pipeline import decontaminate as _decontaminate_colors

# Register <ComfyUI>/models/lucida as a model folder
folder_paths.add_model_folder_path("lucida", os.path.join(folder_paths.models_dir, "lucida"))

_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Single-entry cache: reloading an 885 MB model per run would be painful.
_MODEL_CACHE = {}

MODEL_FILENAME = "model.safetensors"
MODEL_URL = "https://huggingface.co/egeorcun/lucida/resolve/main/model.safetensors"


def _ensure_model_downloaded():
    """Download model.safetensors if not present in models/lucida."""
    model_dir = os.path.join(folder_paths.models_dir, "lucida")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    
    if os.path.isfile(model_path):
        return model_path
    
    print(f"[Star Lucida RMBG] Model not found at {model_path}")
    print(f"[Star Lucida RMBG] Downloading from {MODEL_URL} ...")
    
    try:
        import urllib.request
        import tempfile
        
        # Download to temp file first, then move
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
            tmp_path = tmp_file.name
            
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r[Star Lucida RMBG] Downloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(MODEL_URL, tmp_path, reporthook=_progress)
        print()  # newline after progress
        
        # Move to final location
        import shutil
        shutil.move(tmp_path, model_path)
        
        print(f"[Star Lucida RMBG] Download complete: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"[Star Lucida RMBG] Download failed: {e}")
        raise FileNotFoundError(
            f"[Star Lucida RMBG] Could not download model from {MODEL_URL}. "
            f"Please download manually and place at {model_path}"
        )


def _get_model(device: torch.device, dtype: torch.dtype) -> LucidaModel:
    """Load or retrieve cached Lucida model."""
    model_path = _ensure_model_downloaded()
    
    key = (model_path, str(device), str(dtype))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # keep at most one model resident
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Star Lucida RMBG] Loading {os.path.basename(model_path)} on {device} ({dtype}) ...")
        _MODEL_CACHE[key] = LucidaModel(model_path, device, dtype)
    return _MODEL_CACHE[key]


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model_management.get_torch_device()


class StarLucidaRMBG:
    """Remove background with Lucida BiRefNet fine-tune.
    
    Excels at semi-transparent objects (glass), camouflage, text/logos with
    soft shadows, glow/VFX effects, and illustrations/line art.
    
    Based on: https://github.com/egeorcun/lucida
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (batches processed one by one)"}),
                "refine": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Edge-refinement pass: re-runs the model on the "
                        "uncertain alpha band at higher effective resolution. "
                        "Slower, sharper edges.",
                    },
                ),
                "decontaminate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Clean background colour spill from the RGB channels "
                        "of the cut-out (uses pymatting if installed, otherwise a "
                        "built-in nearest-colour fallback).",
                    },
                ),
                "dtype": (
                    ["fp32", "bf16", "fp16"],
                    {
                        "default": "fp32",
                        "tooltip": "fp32 matches the reference pipeline. bf16/fp16 "
                        "roughly halve VRAM on GPU. CPU always runs fp32.",
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto", "tooltip": "'auto' uses ComfyUI's device."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "mask")
    OUTPUT_TOOLTIPS = (
        "RGBA cut-out (4-channel IMAGE; save as PNG to keep transparency).",
        "Alpha matte as a MASK for compositing.",
    )
    FUNCTION = "remove_background"
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = (
        "Background removal with Lucida (BiRefNet fine-tune by egeorcun). "
        "Keeps glass/transparency, camouflage, text, glow and line art. "
        "Model auto-downloads from HuggingFace on first use."
    )

    def remove_background(self, image, refine, decontaminate, dtype, device):
        torch_device = _resolve_device(device)
        torch_dtype = _DTYPES[dtype] if torch_device.type != "cpu" else torch.float32
        segmenter = _get_model(torch_device, torch_dtype)

        rgba_out, mask_out = [], []
        total = image.shape[0]
        for i in range(total):
            model_management.throw_exception_if_processing_interrupted()
            arr = np.round(image[i].cpu().numpy() * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(arr, mode="RGB")

            alpha = segmenter.predict_alpha(pil_image)
            if refine:
                alpha = refine_alpha(segmenter, pil_image, alpha)

            rgba = _decontaminate_colors(pil_image, alpha) if decontaminate else compose_rgba(pil_image, alpha)

            rgba_out.append(torch.from_numpy(np.asarray(rgba).astype(np.float32) / 255.0))
            mask_out.append(torch.from_numpy(alpha))
            if total > 1:
                print(f"[Star Lucida RMBG] {i + 1}/{total} done")

        return (torch.stack(rgba_out), torch.stack(mask_out))


NODE_CLASS_MAPPINGS = {
    "StarLucidaRMBG": StarLucidaRMBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLucidaRMBG": "⭐ Star Lucida RMBG",
}
