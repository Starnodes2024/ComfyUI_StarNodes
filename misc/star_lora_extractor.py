import json
import math
import os
import torch

import folder_paths
import comfy.utils

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


MODEL_FOLDER = "diffusion_models"


def _get_model_folder_type():
    """Return the folder type to use: diffusion_models (models/diffusion_models)."""
    if MODEL_FOLDER in folder_paths.folder_names_and_paths:
        return MODEL_FOLDER
    # Fallback for installations without the diffusion_models folder registered
    return "checkpoints"


def _get_model_list():
    try:
        return folder_paths.get_filename_list(_get_model_folder_type())
    except Exception:
        return []


# Checkpoint key prefix -> ComfyUI LoRA key prefix (order: most specific first)
_LORA_KEY_PREFIXES = [
    ("conditioner.embedders.0.transformer.", "lora_te1_"),
    ("conditioner.embedders.1.model.transformer.", "lora_te2_"),
    ("model.diffusion_model.", "lora_unet_"),
    ("cond_stage_model.transformer.", "lora_te_"),
]

_TEXT_ENCODER_MARKERS = ("lora_te1_", "lora_te2_", "lora_te_")


def _checkpoint_key_to_lora_key(key):
    """Convert a checkpoint state-dict key to a ComfyUI LoRA base key.

    Single diffusion-model files (models/diffusion_models) use raw keys
    without a prefix; ComfyUI wraps them internally as diffusion_model.*,
    so they map to lora_unet_* just like full checkpoints.
    """
    for prefix, replacement in _LORA_KEY_PREFIXES:
        if key.startswith(prefix):
            key = replacement + key[len(prefix):]
            break
    else:
        key = "lora_unet_" + key
    key = key.replace(".", "_")
    if key.endswith("_weight"):
        key = key[: -len("_weight")]
    return key


_UNSUPPORTED_QUANT_FORMATS = ("nvfp4", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8")

_HADAMARD_CACHE = {}

_H4 = [[1.0, 1.0, 1.0, -1.0],
       [1.0, 1.0, -1.0, 1.0],
       [1.0, -1.0, 1.0, 1.0],
       [-1.0, 1.0, 1.0, 1.0]]


def _build_hadamard(size, device):
    """Normalized orthogonal Hadamard matrix, same construction as comfy_kitchen convrot."""
    cache_key = (size, str(device))
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"convrot group size must be a power of 4, got {size}")
    h4 = torch.tensor(_H4, dtype=torch.float32, device=device)
    h = h4
    n = 4
    while n < size:
        h = torch.kron(h, h4)
        n *= 4
    h = h / math.sqrt(size)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _unrotate_convrot(weight, group_size):
    """Inverse of comfy_kitchen's convrot weight rotation (H is symmetric + orthonormal)."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"convrot: in_features {in_f} not divisible by group_size {group_size}")
    h = _build_hadamard(group_size, weight.device)
    return (weight.reshape(out_f, in_f // group_size, group_size) @ h.T).reshape(out_f, in_f)


def _dequantize_to_float(q, scale, fmt, info, key, device):
    """Restore the real fp32 weight from a ComfyUI-quantized tensor."""
    if fmt in _UNSUPPORTED_QUANT_FORMATS:
        raise ValueError(
            f"'{key}' uses packed quantization '{fmt}' which cannot be diffed directly. "
            "Use both models in bf16/fp16/fp8-scaled or row-wise int8 format."
        )
    w = q.to(device=device, dtype=torch.float32)
    if scale is not None:
        s = scale.to(device=device, dtype=torch.float32)
        if s.numel() == 1:
            w = w * s
        elif s.numel() == q.shape[0]:
            w = w * s.reshape((q.shape[0],) + (1,) * (q.ndim - 1))
        else:
            raise ValueError(f"'{key}': unsupported scale shape {tuple(s.shape)} for quantization '{fmt}'.")
    if info.get("convrot", False):
        if w.ndim != 2:
            raise ValueError(f"'{key}': convrot is only supported for 2D weights.")
        w = _unrotate_convrot(w, int(info.get("convrot_groupsize", 256)))
    return w


class _LazyModelReader:
    """Reads tensors from a checkpoint file on demand, key by key.

    .safetensors files are memory-mapped, so only the tensors currently
    being processed are ever held in RAM (very low memory usage).
    Legacy .ckpt/.pt files have to be loaded fully and need more RAM.
    """

    def __init__(self, path):
        self.path = path
        self._handle = None
        self._state = None
        if safe_open is not None and path.lower().endswith(".safetensors"):
            self._handle = safe_open(path, framework="pt", device="cpu")
            self.keys = set(self._handle.keys())
            self.lazy = True
        else:
            self._state = comfy.utils.load_torch_file(path)
            self.keys = set(self._state.keys())
            self.lazy = False

    def get(self, key):
        if self.lazy:
            return self._handle.get_tensor(key)
        return self._state[key]

    def shape_of(self, key):
        """Tensor shape without loading the data (for lazy readers)."""
        if self.lazy:
            return tuple(self._handle.get_slice(key).get_shape())
        return tuple(self._state[key].shape)

    def get_weight(self, key, device):
        """Return the real fp32 weight, dequantizing ComfyUI-quantized tensors."""
        w = self.get(key)
        blob_key = key[: -len("weight")] + "comfy_quant"
        if blob_key not in self.keys:
            return w.to(device=device, dtype=torch.float32)
        info = json.loads(self.get(blob_key).numpy().tobytes().decode("utf-8"))
        scale_key = key + "_scale"
        scale = self.get(scale_key) if scale_key in self.keys else None
        return _dequantize_to_float(w, scale, info.get("format", ""), info, key, device)


class StarLoraExtractor:
    """
    Extracts the difference between a base model and a merged model
    (base + baked-in LoRAs) and saves it as a single LoRA in native
    ComfyUI format.

    Uses per-matrix SVD decomposition of the weight difference:
        diff = merged_weight - base_weight  ~=  lora_up @ lora_down

    Memory-friendly: safetensors files are read key by key, so the full
    checkpoints are never held in RAM/VRAM at once.
    """

    BGCOLOR = "#4d1212"
    COLOR = "#1d124d"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        ckpt_list = _get_model_list()
        return {
            "required": {
                "base_model": (ckpt_list, {"tooltip": "The clean base diffusion model (from models/diffusion_models)."}),
                "merge_model": (ckpt_list, {"tooltip": "The diffusion model with LoRAs already baked in (from models/diffusion_models)."}),
                "lora_name": ("STRING", {
                    "default": "star_extracted_lora",
                    "tooltip": "File name for the extracted LoRA (saved into models/loras). May contain subfolders."
                }),
                "dim": ("INT", {
                    "default": 16, "min": 1, "max": 1024, "step": 1,
                    "tooltip": "Rank/size of the LoRA for Linear layers. Higher = more detail, bigger file."
                }),
                "conv_dim": ("INT", {
                    "default": 8, "min": 1, "max": 512, "step": 1,
                    "tooltip": "Rank for Conv2d layers (UNet resnet blocks). Lower than dim is usually fine."
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Save precision/quality of the extracted LoRA weights."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Where the SVD math runs. cpu = lowest RAM/VRAM usage, cuda = much faster if it fits."
                }),
            },
            "optional": {
                "extract_convs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also extract Conv2d layer differences (recommended for full fidelity)."
                }),
                "include_text_encoder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also extract text encoder (CLIP) differences."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow overwriting an existing LoRA file with the same name."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "extract"
    DESCRIPTION = """Extracts the difference between a base model and a
merge model (LoRAs baked in) and saves it as a single LoRA
in native ComfyUI format (models/loras folder).

Low-RAM friendly: both checkpoints are read key by key
(safetensors memory-mapping), SVD is done per matrix in fp32.

Use dim (Linear rank) + conv_dim (Conv rank) + precision to
control the size and quality of the extracted LoRA."""

    def extract(self, base_model, merge_model, lora_name, dim, conv_dim,
                precision, device, extract_convs=True,
                include_text_encoder=True, overwrite=False):
        if base_model == merge_model:
            raise ValueError("base_model and merge_model must be different files.")

        device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

        base_path = folder_paths.get_full_path_or_raise(_get_model_folder_type(), base_model)
        merge_path = folder_paths.get_full_path_or_raise(_get_model_folder_type(), merge_model)

        # Resolve the output path (models/loras)
        safe_name = lora_name.strip()
        if not safe_name:
            raise ValueError("lora_name must not be empty.")
        if not safe_name.lower().endswith(".safetensors"):
            safe_name += ".safetensors"
        existing = folder_paths.get_full_path("loras", safe_name)
        if existing is not None and not overwrite:
            raise FileExistsError(
                f"A LoRA named '{safe_name}' already exists in models/loras. "
                "Choose another name or enable 'overwrite'."
            )
        loras_dir = folder_paths.folder_names_and_paths["loras"][0][0]
        save_path = os.path.join(loras_dir, safe_name)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        base = _LazyModelReader(base_path)
        merge = _LazyModelReader(merge_path)

        common_keys = sorted(base.keys & merge.keys)
        print(f"[Star LoRA Extractor] base ({os.path.basename(base_path)}): {len(base.keys)} keys "
              f"({'lazy' if base.lazy else 'full'} load)")
        print(f"[Star LoRA Extractor] merge ({os.path.basename(merge_path)}): {len(merge.keys)} keys "
              f"({'lazy' if merge.lazy else 'full'} load)")
        if not (base.lazy and merge.lazy):
            print("[Star LoRA Extractor] Warning: non-safetensors checkpoint detected - "
                  "whole file(s) had to be loaded into RAM.")

        lora_state = {}
        stats = {"extracted": 0, "unchanged": 0, "skipped_shape": 0,
                 "skipped_conv": 0, "skipped_te": 0, "skipped_vae": 0}

        progress = comfy.utils.ProgressBar(len(common_keys)) if hasattr(comfy.utils, "ProgressBar") else None

        for i, key in enumerate(common_keys):
            if progress is not None:
                progress.update_absolute(i + 1)

            try:
                if key.startswith("first_stage_model."):
                    stats["skipped_vae"] += 1
                    continue
                if not key.endswith("weight"):
                    continue

                lora_base_key = _checkpoint_key_to_lora_key(key)
                if not include_text_encoder and lora_base_key.startswith(_TEXT_ENCODER_MARKERS):
                    stats["skipped_te"] += 1
                    continue

                shape = base.shape_of(key)
                if shape != merge.shape_of(key) or len(shape) not in (2, 4):
                    stats["skipped_shape"] += 1
                    continue

                if len(shape) == 4 and not extract_convs:
                    stats["skipped_conv"] += 1
                    continue

                # fp32 diff on the chosen device (dequantizing quantized weights)
                t_base = base.get_weight(key, device)
                t_merge = merge.get_weight(key, device)
                diff = t_merge - t_base
                del t_base, t_merge

                if float(diff.abs().max()) < 1e-7:
                    stats["unchanged"] += 1
                    del diff
                    continue

                if diff.ndim == 4 and (diff.shape[2] == 1 and diff.shape[3] == 1):
                    # 1x1 conv: treat as Linear, keep 4D shapes for the save
                    rank = min(dim, diff.shape[0], diff.shape[1])
                    up, down = _svd_decompose(diff[:, :, 0, 0], rank)
                    lora_up = up.reshape(up.shape[0], up.shape[1], 1, 1)
                    lora_down = down.reshape(down.shape[0], down.shape[1], 1, 1)
                elif diff.ndim == 4:
                    # conv with kernel size > 1: factorize (out, in*kh*kw)
                    out_c, in_c, kh, kw = diff.shape
                    rank = min(conv_dim, out_c, in_c * kh * kw)
                    up, down = _svd_decompose(diff.reshape(out_c, in_c * kh * kw), rank)
                    lora_up = up.reshape(out_c, rank, 1, 1)
                    lora_down = down.reshape(rank, in_c, kh, kw)
                else:
                    rank = min(dim, diff.shape[0], diff.shape[1])
                    up, down = _svd_decompose(diff, rank)
                    lora_up, lora_down = up, down
                del diff, up, down

                lora_state[f"{lora_base_key}.lora_up.weight"] = lora_up.to(dtype=save_dtype).cpu().contiguous()
                lora_state[f"{lora_base_key}.lora_down.weight"] = lora_down.to(dtype=save_dtype).cpu().contiguous()
                lora_state[f"{lora_base_key}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
                del lora_up, lora_down
                stats["extracted"] += 1

                if device == "cuda" and stats["extracted"] % 200 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                raise RuntimeError(f"[Star LoRA Extractor] failed on key '{key}': {e}") from e

        if not lora_state:
            raise ValueError("No differences between the two models were found - LoRA would be empty.")

        save_file(lora_state, save_path, metadata={"format": "pt"})

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        info = (
            "=== Star LoRA Extractor ===\n"
            f"Base model:  {base_model}\n"
            f"Merge model: {merge_model}\n"
            f"Saved LoRA:  {save_path} ({file_size_mb:.1f} MB)\n"
            f"Precision:   {precision} | device: {device}\n"
            f"Ranks:       linear dim={dim}, conv dim={conv_dim}\n"
            f"\n"
            f"Layers extracted:        {stats['extracted']}\n"
            f"Unchanged (skipped):     {stats['unchanged']}\n"
            f"Skipped (VAE):           {stats['skipped_vae']}\n"
            f"Skipped (text encoder):  {stats['skipped_te']}\n"
            f"Skipped (convs off):     {stats['skipped_conv']}\n"
            f"Skipped (shape/type):    {stats['skipped_shape']}\n"
            f"\n"
            f"Load it with the standard ComfyUI 'Load LoRA' node (filename: {os.path.basename(save_path)})."
        )
        print(f"[Star LoRA Extractor] {info}")
        return (info,)


def save_file(state_dict, path, metadata=None):
    from safetensors.torch import save_file as _st_save_file
    _st_save_file(state_dict, path, metadata=metadata)


def _svd_decompose(diff, rank):
    """Factorize a 2D diff matrix into lora_up @ lora_down of the given rank."""
    U, S, V = torch.linalg.svd(diff)
    rank = min(rank, S.shape[0])
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    lora_up = U @ torch.diag(S)  # (out, rank)
    lora_down = V                # (rank, in)
    return lora_up, lora_down


NODE_CLASS_MAPPINGS = {
    "StarLoraExtractor": StarLoraExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLoraExtractor": "⭐ Star LoRA Extractor",
}
