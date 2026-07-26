"""
⭐ Star Show Everything

A ComfyUI utility node that accepts any input type, previews it,
and passes it through unchanged as an ANY / wildcard output.
Also provides a STRING info output with a human-readable summary
of the connected value (model names, tensor shapes, ints, etc.).
"""

import os
import uuid
import traceback

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


class AnyType(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


ANY = AnyType("*")

TEMP_SUBFOLDER = "star_show_everything"
MAX_TEXT = 4000
MAX_PREVIEW_IMAGES = 4

NAME_ATTRS = [
    "__name__",
    "__qualname__",
    "model_name",
    "clip_name",
    "vae_name",
    "lora_name",
    "embedding_name",
    "style_model_name",
    "control_net_name",
    "name",
    "display_name",
    "filename",
    "file_name",
    "path",
    "model_path",
    "ckpt_name",
    "ckpt_path",
    "unet_name",
    "unet_path",
    "model_file",
    "full_path",
    "checkpoint_name",
    "model_id",
    "repo_id",
]

NESTED_ATTRS = [
    "model",
    "patcher",
    "cond_stage_model",
    "model_config",
    "clip",
    "vae",
    "control_model",
    "style_model",
    "gligen",
    "conditioner",
    "tokenizer",
    "unet_config",
    "clip_config",
    "vae_config",
    "config",
]


def _safe_str(value, limit=MAX_TEXT):
    try:
        text = str(value)
    except Exception:
        try:
            text = repr(value)
        except Exception:
            return "<unprintable value>"

    if len(text) > limit:
        return text[:limit] + f"\n... [{len(text) - limit} more characters]"
    return text


def _safe_attr_text(value, limit=160):
    try:
        if isinstance(value, (str, int, float, bool, complex, type(None))):
            return _safe_str(value, limit)

        if isinstance(value, os.PathLike):
            try:
                return _safe_str(os.fspath(value), limit)
            except Exception:
                pass

        text = object.__repr__(value)
        if len(text) > limit:
            text = text[:limit] + "..."
        return text
    except Exception:
        return "<object>"


def _name_text(value):
    try:
        if isinstance(value, (str, bytes)):
            text = value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
            text = text.strip()
            return text or None

        if hasattr(value, "__fspath__"):
            text = str(os.fspath(value)).strip()
            return text or None
    except Exception:
        pass

    return None


def _clean_name(name):
    try:
        name = str(name).strip()
        if not name:
            return None

        if "/" in name or "\\" in name:
            base = os.path.basename(name.replace("\\", "/"))
            if base and base != name:
                if len(name) <= 160:
                    return f"{base}\n{name}"
                return base

        return name
    except Exception:
        return None


def _find_name(obj, depth=0, visited=None):
    if visited is None:
        visited = set()

    if depth > 4 or obj is None:
        return None

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if isinstance(obj, (bool, int, float, complex, type(None))):
        return None

    if isinstance(obj, (str, bytes)):
        return _name_text(obj)

    if isinstance(obj, os.PathLike):
        return _name_text(obj)

    if torch and torch.is_tensor(obj):
        return None

    if np and isinstance(obj, np.ndarray):
        return None

    if Image and isinstance(obj, Image.Image):
        return None

    for attr in NAME_ATTRS:
        try:
            text = _name_text(getattr(obj, attr, None))
            if text:
                return text
        except Exception:
            pass

    if isinstance(obj, dict):
        for attr in NAME_ATTRS:
            try:
                text = _name_text(obj.get(attr))
                if text:
                    return text
            except Exception:
                pass

    for nested in NESTED_ATTRS:
        try:
            nested_obj = getattr(obj, nested, None)
            if nested_obj is not None:
                found = _find_name(nested_obj, depth + 1, visited)
                if found:
                    return found
        except Exception:
            pass

    if isinstance(obj, dict):
        count = 0
        for value in obj.values():
            found = _find_name(value, depth + 1, visited)
            if found:
                return found

            count += 1
            if count >= 8:
                break

    return None


def _is_image_tensor(t):
    if not torch or not torch.is_tensor(t):
        return False

    try:
        if t.ndim == 3:
            return int(t.shape[-1]) in (1, 3, 4) or int(t.shape[0]) in (1, 3)

        if t.ndim == 4:
            return int(t.shape[-1]) in (1, 3, 4) or int(t.shape[1]) in (1, 3)
    except Exception:
        return False

    return False


def _is_image_np(a):
    if not np or not isinstance(a, np.ndarray):
        return False

    try:
        if a.ndim == 2:
            return True

        if a.ndim == 3:
            return int(a.shape[-1]) in (1, 3, 4) or int(a.shape[0]) in (1, 3)

        if a.ndim == 4:
            return int(a.shape[-1]) in (1, 3, 4) or int(a.shape[1]) in (1, 3)
    except Exception:
        return False

    return False


def _is_conditioning(value):
    if not torch or not isinstance(value, (list, tuple)) or len(value) == 0:
        return False

    first = value[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return False

    cond = first[0]
    meta = first[1]

    if not torch.is_tensor(cond) or not isinstance(meta, dict):
        return False

    conditioning_keys = (
        "pooled_output",
        "width",
        "height",
        "crop_w",
        "crop_h",
        "target_width",
        "target_height",
        "hooks",
        "prompt_type",
        "strength",
        "original_size",
    )

    if any(k in meta for k in conditioning_keys):
        return True

    try:
        if cond.ndim >= 2 and cond.ndim <= 3:
            return True
    except Exception:
        pass

    return False


def _tensor_to_preview_images(tensor):
    if not torch or not torch.is_tensor(tensor):
        return []

    images = []

    try:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            return []

        batch = min(tensor.shape[0], MAX_PREVIEW_IMAGES)
        for i in range(batch):
            img = tensor[i].cpu()

            if img.shape[-1] == 1:
                img = img.repeat(1, 1, 3)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]

            img = img.clamp(0, 1)
            img_np = (img.numpy() * 255).astype("uint8")

            if Image is not None:
                pil_img = Image.fromarray(img_np)
                images.append(pil_img)
    except Exception:
        pass

    return images


def _np_to_preview_images(arr):
    if not np or not isinstance(arr, np.ndarray):
        return []

    images = []

    try:
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
            arr = arr[None]

        if arr.ndim == 3:
            arr = arr[None]

        if arr.ndim != 4:
            return []

        batch = min(arr.shape[0], MAX_PREVIEW_IMAGES)
        for i in range(batch):
            img = arr[i]
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            elif img.shape[-1] == 1:
                img = img.repeat(3, axis=2)

            if img.dtype != "uint8":
                img = (img * 255).clip(0, 255).astype("uint8")

            if Image is not None:
                pil_img = Image.fromarray(img)
                images.append(pil_img)
    except Exception:
        pass

    return images


def _save_preview_images(images):
    if not images or folder_paths is None:
        return []

    results = []
    temp_dir = folder_paths.get_temp_directory()
    sub_dir = os.path.join(temp_dir, TEMP_SUBFOLDER)
    os.makedirs(sub_dir, exist_ok=True)

    for pil_img in images:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(sub_dir, filename)
        try:
            pil_img.save(filepath)
            results.append({
                "filename": filename,
                "subfolder": TEMP_SUBFOLDER,
                "type": "temp",
            })
        except Exception:
            pass

    return results


def _build_info_text(value):
    lines = []

    type_name = type(value).__name__
    lines.append(f"Type: {type_name}")

    name = _find_name(value)
    if name:
        clean = _clean_name(name)
        if clean:
            lines.append(f"Name: {clean}")

    if torch and torch.is_tensor(value):
        lines.append(f"Shape: {tuple(value.shape)}")
        lines.append(f"Dtype: {value.dtype}")
        lines.append(f"Device: {value.device}")
        if value.numel() <= 8:
            lines.append(f"Values: {value.tolist()}")
        else:
            lines.append(f"Min: {value.min().item():.6f}")
            lines.append(f"Max: {value.max().item():.6f}")
            lines.append(f"Mean: {value.float().mean().item():.6f}")

    elif np and isinstance(value, np.ndarray):
        lines.append(f"Shape: {value.shape}")
        lines.append(f"Dtype: {value.dtype}")
        if value.size <= 8:
            lines.append(f"Values: {value.tolist()}")

    elif isinstance(value, (list, tuple)):
        lines.append(f"Length: {len(value)}")
        if len(value) > 0:
            lines.append(f"Item[0] type: {type(value[0]).__name__}")
            if _is_conditioning(value):
                lines.append("Detected: CONDITIONING")
                try:
                    meta = value[0][1]
                    if isinstance(meta, dict):
                        for k, v in meta.items():
                            if k in ("pooled_output", "width", "height", "crop_w", "crop_h",
                                     "target_width", "target_height", "strength"):
                                if torch and torch.is_tensor(v):
                                    lines.append(f"  {k}: tensor {tuple(v.shape)}")
                                else:
                                    lines.append(f"  {k}: {v}")
                except Exception:
                    pass

    elif isinstance(value, dict):
        lines.append(f"Keys: {len(value)}")
        for k in list(value.keys())[:12]:
            v = value[k]
            vtype = type(v).__name__
            if torch and torch.is_tensor(v):
                lines.append(f"  {k}: {vtype} {tuple(v.shape)}")
            elif isinstance(v, (str, int, float, bool)):
                lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {k}: {vtype}")

    elif isinstance(value, (int, float, bool)):
        lines.append(f"Value: {value}")

    elif isinstance(value, str):
        if len(value) <= 200:
            lines.append(f"Value: {value}")
        else:
            lines.append(f"Value: {value[:200]}... ({len(value)} chars)")

    elif value is None:
        lines.append("Value: None")

    else:
        text = _safe_attr_text(value, 300)
        lines.append(f"Repr: {text}")

        if torch:
            for attr in ("model", "patcher"):
                sub = getattr(value, attr, None)
                if sub is not None:
                    sub_name = _find_name(sub)
                    if sub_name:
                        lines.append(f"  {attr} name: {_clean_name(sub_name)}")

    return "\n".join(lines)


class StarShowEverything:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY, {"tooltip": "Connect any data type — MODEL, IMAGE, LATENT, CONDITIONING, STRING, INT, FLOAT, etc. The value is previewed and passed through unchanged."}),
            },
        }

    RETURN_TYPES = (ANY, "STRING", "STRING")
    RETURN_NAMES = ("anything", "info", "name")
    OUTPUT_TOOLTIPS = (
        "The connected value, passed through unchanged.",
        "Human-readable summary of the value (type, shape, dtype, model name, etc.).",
        "Best-effort name/label extracted from the value (e.g. model filename).",
    )
    FUNCTION = "show"
    DESCRIPTION = (
        "Universal debug and inspection node. Connect any output (MODEL, IMAGE, "
        "LATENT, CONDITIONING, STRING, INT, FLOAT, MASK, etc.) to see a "
        "human-readable summary — type, shape, dtype, device, model names, tensor "
        "stats, and more. Image tensors are previewed inline. The value is passed "
        "through unchanged so you can insert it anywhere in your workflow."
    )

    def show(self, anything):
        info_text = _build_info_text(anything)

        name = _find_name(anything)
        name_text = _clean_name(name) if name else ""

        ui_result = {"ui": {"text": [info_text]}}

        preview_images = []

        if torch and torch.is_tensor(anything) and _is_image_tensor(anything):
            preview_images = _tensor_to_preview_images(anything)
        elif np and isinstance(anything, np.ndarray) and _is_image_np(anything):
            preview_images = _np_to_preview_images(anything)
        elif Image and isinstance(anything, Image.Image):
            preview_images = [anything.copy()]

        if preview_images:
            saved = _save_preview_images(preview_images)
            if saved:
                ui_result["ui"]["images"] = saved

        return {
            "result": (anything, info_text, name_text),
            "ui": ui_result["ui"],
        }


NODE_CLASS_MAPPINGS = {
    "StarShowEverything": StarShowEverything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarShowEverything": "⭐ Star Show Everything",
}
