"""
StarNodes MiniMax-H3 merge utilities.

Low-RAM, streaming safetensors tooling for MiniMax-H3 (H3-Omni-Transformer)
checkpoints as distributed by Comfy-Org/MiniMax-H3 (single-file diffusion models:
bf16 / fp16 / fp32 / fp8_scaled / int8 variants, full or AdaLN-pruned).

Everything here works tensor-by-tensor: merging two 62 GiB checkpoints does NOT
require holding either model fully in RAM (peak ~= largest single tensor).
"""

import json
import os
import struct
import uuid

import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Optional ComfyUI integration (this module must also import standalone for tests)
# ---------------------------------------------------------------------------
try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - outside ComfyUI
    folder_paths = None


def log(msg):
    print(f"[Star Minimax H3] {msg}")


# ---------------------------------------------------------------------------
# dtype tables
# ---------------------------------------------------------------------------
TORCH_TO_ST = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
if hasattr(torch, "float8_e4m3fn"):
    TORCH_TO_ST[torch.float8_e4m3fn] = "F8_E4M3"
if hasattr(torch, "float8_e5m2"):
    TORCH_TO_ST[torch.float8_e5m2] = "F8_E5M2"

ST_TO_TORCH = {v: k for k, v in TORCH_TO_ST.items()}

OUT_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

FLOATY_ST = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


class UnsupportedQuantError(Exception):
    """Raised when a tensor uses a quant format we cannot dequantize inline."""


# ---------------------------------------------------------------------------
# File inspection
# ---------------------------------------------------------------------------
class FileInfo:
    """Header-only view of a safetensors file."""

    def __init__(self, path):
        self.path = path
        self.keys = {}          # key -> (st_dtype_str, shape)
        self.metadata = {}
        with safe_open(path, framework="pt") as f:
            md = f.metadata()
            if md:
                self.metadata = dict(md)
            for k in f.keys():
                sl = f.get_slice(k)
                dt = sl.get_dtype()
                self.keys[k] = (TORCH_TO_ST.get(dt, str(dt)), tuple(sl.get_shape()))

    def has(self, key):
        return key in self.keys

    def dtype_str(self, key):
        return self.keys[key][0]

    def shape(self, key):
        return self.keys[key][1]


# ---------------------------------------------------------------------------
# Quantization handling (ComfyUI native .comfy_quant format)
# ---------------------------------------------------------------------------
def _decode_comfy_quant(tensor):
    try:
        raw = bytes(tensor.view(torch.uint8).tolist()) if tensor.dim() else b""
        raw = bytes(b for b in raw if b != 0)
        return json.loads(raw.decode("utf-8", "ignore") or "{}")
    except Exception:
        return {}


def build_quant_index(info):
    """
    Map weight key -> {"format": str, "scale_keys": {...}} for ComfyUI-native
    quantized checkpoints (fp8_scaled, int8_*, ...).
    """
    quant = {}
    for k in list(info.keys):
        if not k.endswith(".comfy_quant"):
            continue
        base = k[: -len(".comfy_quant")]
        # the marker may sit on the weight itself or on the layer prefix
        if base in info.keys:
            wkey = base
        elif base + ".weight" in info.keys:
            wkey = base + ".weight"
        else:
            continue
        with safe_open(info.path, framework="pt") as f:
            marker = _decode_comfy_quant(f.get_tensor(k))
        prefix = wkey[: -len(".weight")] if wkey.endswith(".weight") else wkey
        scales = {}
        for cand, role in (
            (prefix + ".weight_scale", "weight_scale"),
            (prefix + ".scale_weight", "weight_scale"),
            (wkey + ".scale", "weight_scale"),
            (prefix + ".input_scale", "input_scale"),
            (prefix + ".pre_quant_scale", "pre_quant_scale"),
        ):
            if cand in info.keys:
                scales[role] = cand
        quant[wkey] = {
            "format": str(marker.get("format", "")).lower(),
            "scale_keys": scales,
            "marker_key": k,
        }
    return quant


def dequantize(f, key, qinfo):
    """
    Read `key` from open handle `f` and return a float32 tensor, transparently
    dequantizing ComfyUI-native quantized storage.

    Supported inline: float8_e4m3fn / float8_e5m2 (fp8_scaled), int8 tensorwise
    and int8 row-wise. Rotation-based formats (convrot) need the full ComfyUI
    runtime -> UnsupportedQuantError, handled by the caller via comfy fallback.
    """
    t = f.get_tensor(key)
    if qinfo is None:
        return t.to(torch.float32) if TORCH_TO_ST.get(t.dtype, "") in FLOATY_ST else t

    fmt = qinfo.get("format", "")
    scales = qinfo.get("scale_keys", {})

    def _scale():
        sk = scales.get("weight_scale")
        if sk is None:
            raise UnsupportedQuantError(f"quantized tensor {key} has no weight_scale")
        s = f.get_tensor(sk).to(torch.float32)
        return s

    if fmt.startswith("float8"):
        return t.to(torch.float32) * _scale()
    if "int8" in fmt and "convrot" not in fmt:
        s = _scale()
        # row-wise scales come as (out, 1) or (out,)
        if s.dim() == 1 and t.dim() == 2 and s.shape[0] == t.shape[0]:
            s = s.unsqueeze(1)
        return t.to(torch.float32) * s
    # convrot / nvfp4 / mxfp8 / anything exotic -> needs comfy-kitchen runtime
    raise UnsupportedQuantError(f"format '{fmt or 'unknown'}' on {key}")


def load_via_comfy_dequantized(path):
    """
    Fallback for rotation/block quant formats (int8_convrot, nvfp4, ...):
    let ComfyUI itself parse the file and dequantize every QuantizedTensor.
    Returns a dict key -> float32 tensor. Uses as much RAM as the dequantized
    model (bf16-equivalent ~2x file size), so only used when needed.
    """
    try:
        from comfy.utils import load_torch_file  # type: ignore
    except Exception as e:  # pragma: no cover
        raise UnsupportedQuantError(
            "This checkpoint uses a quantization format that can only be "
            "dequantized by the ComfyUI runtime (comfy-kitchen), which could "
            f"not be imported: {e}"
        )
    log(f"using ComfyUI runtime to dequantize {os.path.basename(path)} "
        "(higher RAM usage, one-time)...")
    sd = load_torch_file(path, safe_load=True)
    out = {}
    for k, v in sd.items():
        if k.endswith(".comfy_quant"):
            continue
        if hasattr(v, "dequantize"):           # comfy QuantizedTensor
            out[k] = v.dequantize().to(torch.float32)
        elif hasattr(v, "_params"):            # defensive: other quant wrappers
            try:
                out[k] = v.dequantize().to(torch.float32)
            except Exception:
                out[k] = torch.as_tensor(v).to(torch.float32)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(torch.float32) if v.is_floating_point() or v.dtype in (
                getattr(torch, "float8_e4m3fn", torch.int8),
                getattr(torch, "float8_e5m2", torch.int8),
            ) else v
    return out


# ---------------------------------------------------------------------------
# Tensor sources (streaming file or in-memory dict share one interface)
# ---------------------------------------------------------------------------
class TensorSource:
    """Uniform lazy access: .keys() and .get(key)->float32 tensor."""

    def __init__(self, path, force_comfy=False):
        self.path = path
        self.info = FileInfo(path)
        self.quant = build_quant_index(self.info)
        self._dict = None
        self._handle = None
        if force_comfy or any(
            ("convrot" in q["format"] or "nvfp4" in q["format"] or "mxfp" in q["format"])
            for q in self.quant.values()
        ):
            self._dict = load_via_comfy_dequantized(path)

    def keys(self):
        if self._dict is not None:
            return set(self._dict.keys())
        return set(self.info.keys) - {
            k for k in self.info.keys
            if k.endswith(".comfy_quant")
            or k.endswith(".weight_scale") or k.endswith(".scale_weight")
            or k.endswith(".input_scale") or k.endswith(".pre_quant_scale")
        }

    def dtype_str(self, key):
        if self._dict is not None:
            return "F32" if self._dict[key].is_floating_point() else TORCH_TO_ST.get(
                self._dict[key].dtype, "I64")
        if key in self.quant:
            return "F32"  # dequantizes to float
        return self.info.dtype_str(key)

    def shape(self, key):
        if self._dict is not None:
            return tuple(self._dict[key].shape)
        return self.info.shape(key)

    def get(self, key):
        if self._dict is not None:
            v = self._dict[key]
            return v if not v.is_floating_point() else v.to(torch.float32)
        if self._handle is None:
            self._handle = safe_open(self.path, framework="pt")
        return dequantize(self._handle, key, self.quant.get(key))

    def close(self):
        if self._handle is not None:
            try:
                self._handle.__exit__(None, None, None)
            except Exception:
                pass
            self._handle = None
        self._dict = None


# ---------------------------------------------------------------------------
# Streaming safetensors writer
# ---------------------------------------------------------------------------
def _tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if TORCH_TO_ST.get(t.dtype, "").startswith("F8_"):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def write_safetensors_stream(path, plan, producer, metadata=None):
    """
    plan:     list of (key, torch_dtype, shape) in write order
    producer: callable(key) -> torch.Tensor (any float dtype; cast here)
    Writes a valid safetensors file without holding more than one tensor in RAM.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = {}
    offset = 0
    for key, dt, shape in plan:
        n = 1
        for s in shape:
            n *= s
        n *= torch.tensor([], dtype=dt).element_size()
        header[key] = {
            "dtype": TORCH_TO_ST[dt],
            "shape": [int(s) for s in shape],
            "data_offsets": [offset, offset + n],
        }
        offset += n
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}

    hjson = json.dumps(header).encode("utf-8")
    pad = (8 - (len(hjson) % 8)) % 8
    hjson += b" " * pad

    tmp = path + ".partial"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        for key, dt, shape in plan:
            t = producer(key)
            if t.dtype != dt:
                t = t.to(dt)
            fh.write(_tensor_bytes(t))
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# MiniMax-H3 architecture guards
# ---------------------------------------------------------------------------
def detect_variant(name, info):
    """Best-effort FL2VA / Ref2VA / pruned detection from filename + keys."""
    low = os.path.basename(name).lower()
    variant = "fl2va" if "fl2va" in low else ("ref2va" if "ref2va" in low else None)
    if variant is None:
        ks = list(info.keys)
        joined = " ".join(ks[:2000]).lower()
        if "ref_" in joined or "_ref" in joined or "reference" in joined:
            variant = "ref2va?"
    pruned = "pruned" in low
    if not pruned:
        # pruned checkpoints replace AdaLN branches with curve tables
        adaln = sum(1 for k in info.keys if "adaln" in k.lower()
                    or "modulation" in k.lower())
        tables = sum(1 for k in info.keys if "curve" in k.lower()
                     or "table" in k.lower())
        if tables > 0 and adaln == 0:
            pruned = True
    return variant, pruned


def check_pair_compatible(src_a, src_b, strict, logfn=log):
    """
    Architecture-respecting sanity checks before merging two H3 checkpoints.
    Returns report dict. Raises on hard incompatibilities.
    """
    ia, ib = src_a.info, src_b.info
    va, pa = detect_variant(src_a.path, ia)
    vb, pb = detect_variant(src_b.path, ib)
    report = {"variant_a": va, "variant_b": vb, "pruned_a": pa, "pruned_b": pb}

    if va and vb and va != vb:
        raise ValueError(
            f"Refusing to merge different H3 DiT variants: "
            f"A looks like '{va}' and B looks like '{vb}'. FL2VA and Ref2VA are "
            "separately trained backbones - merging them corrupts the model."
        )
    if pa != pb:
        raise ValueError(
            "One checkpoint is AdaLN-pruned and the other is full. Their state "
            "dicts are structurally different (the pruned file replaces the ~13B "
            "AdaLN branch parameters with curve tables). Merge pruned with pruned "
            "or full with full."
        )

    ka, kb = src_a.keys(), src_b.keys()
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    shared = sorted(ka & kb)
    mismatched = [k for k in shared if src_a.shape(k) != src_b.shape(k)]

    report.update(
        shared=len(shared), only_a=len(only_a), only_b=len(only_b),
        shape_mismatch=len(mismatched),
    )
    if (only_a or only_b or mismatched) and strict:
        sample = (only_a + only_b + mismatched)[:10]
        raise ValueError(
            "strict_architecture is ON and the two state dicts differ: "
            f"{len(only_a)} keys only in A, {len(only_b)} only in B, "
            f"{len(mismatched)} shape mismatches. First keys: {sample}"
        )
    if mismatched:
        logfn(f"WARNING: {len(mismatched)} shared keys have different shapes; "
              "those will be taken from model A.")
    return report, shared, only_a, only_b, mismatched


# ---------------------------------------------------------------------------
# LoRA helpers (ComfyUI/Kohya-style lora_up / lora_down / alpha)
# ---------------------------------------------------------------------------
_LORA_SUFFIXES = [
    (".lora_up.weight", "up"), (".lora_down.weight", "down"),
    (".lora_B.weight", "up"), (".lora_A.weight", "down"),
    (".lora_up", "up"), (".lora_down", "down"),
    (".alpha", "alpha"),
]


def parse_lora(sd):
    """Split a LoRA state dict into pair groups and leftover tensors."""
    pairs, others = {}, {}
    for k, v in sd.items():
        matched = False
        for suf, role in _LORA_SUFFIXES:
            if k.endswith(suf):
                base = k[: -len(suf)]
                pairs.setdefault(base, {})[role] = v
                matched = True
                break
        if not matched:
            others[k] = v
    complete = {b: p for b, p in pairs.items() if "up" in p and "down" in p}
    dangling = {b: p for b, p in pairs.items() if b not in complete}
    return complete, dangling, others


def lora_delta(pair):
    """dW for one LoRA pair in float32 (alpha/r scaling included)."""
    up = pair["up"].to(torch.float32)
    down = pair["down"].to(torch.float32)
    alpha = pair.get("alpha")
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.flatten()[0])
    r = down.shape[0]
    scale = (alpha / r) if alpha else 1.0
    if down.dim() == 4:  # conv LoRA
        dw = torch.einsum("or,rixy->oixy", up.squeeze(-1).squeeze(-1), down)
    else:
        dw = up @ down
    return dw * scale


def svd_recompose(dw, rank):
    """dW -> (up, down) with given rank via (low-rank) SVD."""
    m, n = dw.shape
    rank = int(max(1, min(rank, m, n)))
    big = max(m, n) > 2048
    if big and rank < min(m, n):
        u, s, v = torch.svd_lowrank(dw, q=min(rank + 8, min(m, n)), niter=2)
        u, s, v = u[:, :rank], s[:rank], v[:, :rank]
        vh = v.t().contiguous()
    else:
        u, s, vh = torch.linalg.svd(dw, full_matrices=False)
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]
    sq = s.clamp_min(0).sqrt()
    up = u * sq.unsqueeze(0)
    down = sq.unsqueeze(1) * vh
    return up.contiguous(), down.contiguous()


def normalize_lora_base(base):
    """Strip known LoRA prefixes -> bare dot-path model key (no .weight)."""
    k = base
    for p in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if k.startswith(p):
            k = k[len(p):]
            break
    return k


def match_lora_to_model(lora_bases, model_keys):
    """
    Map LoRA base key -> model weight key. Handles ComfyUI dot-path keys,
    'diffusion_model.'-prefixed keys and (fallback) kohya lora_unet_ keys.
    Returns (mapping, unmatched_bases).
    """
    norm_to_full = {}
    for mk in model_keys:
        if not mk.endswith(".weight"):
            continue
        core = mk[: -len(".weight")]
        for p in ("model.diffusion_model.", "diffusion_model."):
            if core.startswith(p):
                core = core[len(p):]
                break
        norm_to_full[core] = mk

    mapping, unmatched = {}, []
    for base in lora_bases:
        nb = normalize_lora_base(base)
        if nb in norm_to_full:
            mapping[base] = norm_to_full[nb]
            continue
        if nb.startswith("lora_unet_"):  # kohya underscore fallback
            cand = nb[len("lora_unet_"):].replace("_", ".")
            if cand in norm_to_full:
                mapping[base] = norm_to_full[cand]
                continue
        unmatched.append(base)
    return mapping, unmatched


# ---------------------------------------------------------------------------
# Temp handles
# ---------------------------------------------------------------------------
def temp_path(prefix="star_minimax_h3", ext=".safetensors"):
    if folder_paths is not None:
        d = os.path.join(folder_paths.get_temp_directory(), "star_minimax_h3")
    else:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "star_minimax_h3")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{prefix}_{uuid.uuid4().hex[:12]}{ext}")


def resolve_save_path(location, filename):
    """
    location is the saver node's dropdown label:
    'loras folder' | 'diffusion_models folder' | 'output folder' | 'custom path'.
    """
    filename = filename.strip()
    if location == "custom path" or os.path.isabs(filename):
        return filename if os.path.isabs(filename) else os.path.abspath(filename)
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    if folder_paths is not None:
        if location == "output folder":
            return os.path.join(folder_paths.get_output_directory(), filename)
        kind = "loras" if location == "loras folder" else "diffusion_models"
        base = folder_paths.get_folder_paths(kind)[0]
        return os.path.join(base, filename)
    return os.path.abspath(filename)
