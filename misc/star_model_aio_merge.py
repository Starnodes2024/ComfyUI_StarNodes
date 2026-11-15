import gc
import json
import os
import time

import torch

import folder_paths
import comfy.utils
from comfy.quant_ops import QuantizedTensor


NONE = "none"

FP8_DTYPES = [
    dt for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
]


def _fmt_size(num_bytes):
    gb = num_bytes / (1024 * 1024 * 1024)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _model_sd(model):
    """Diffusion weights of a connected MODEL as prefix-free state dict on CPU."""
    sd = {}
    for k, v in model.model_state_dict("diffusion_model.").items():
        sd[k[len("diffusion_model."):]] = v
    return sd


QUANT_HELPER_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".scale_input",
                         ".scale_weight", ".comfy_quant", ".pre_quant_scale", "scaled_fp8")


def _real_weight(t):
    """Dequantized float32 copy of a weight tensor, or None for non-mergeable
    entries (quantization helpers, integer buffers)."""
    if isinstance(t, QuantizedTensor):
        return t.dequantize().detach().to("cpu", torch.float32)
    if not torch.is_floating_point(t):
        return None
    return t.detach().to("cpu", torch.float32)


def _is_quantized(t):
    return isinstance(t, QuantizedTensor) or t.dtype in FP8_DTYPES


def _apply_lora(merged_sd, lora_path, strength, log):
    """Applies a standard LoRA (kohya down/up or diffusers lora_A/lora_B) onto a
    merged, prefix-free diffusion state dict. Returns (applied, skipped)."""
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

    # normalized (dots and underscores squashed) lookup of mergeable weight keys
    weight_lookup = {}
    for k in merged_sd:
        if k.endswith(".weight"):
            weight_lookup[k[:-len(".weight")].replace(".", "_")] = k

    pairs = {}
    skipped_te = 0
    for k in lora_sd:
        if k.endswith(".lora_down.weight"):
            pairs[k[:-len(".lora_down.weight")]] = (".lora_down.weight", ".lora_up.weight")
        elif k.endswith(".lora_A.weight"):
            pairs[k[:-len(".lora_A.weight")]] = (".lora_A.weight", ".lora_B.weight")

    applied = 0
    skipped = 0
    for base, (down_suffix, up_suffix) in pairs.items():
        if base.startswith("lora_te"):
            skipped_te += 1
            continue

        base_key = base
        for p in ("lora_unet_", "transformer.", "diffusion_model.", "model.diffusion_model."):
            if base_key.startswith(p):
                base_key = base_key[len(p):]

        weight_key = weight_lookup.get(base_key.replace(".", "_"))
        down = lora_sd.get(base + down_suffix)
        up = lora_sd.get(base + up_suffix)
        if down is None or up is None or down.dim() != 2 or up.dim() != 2:
            skipped += 1
            continue
        if weight_key is None or not torch.is_floating_point(merged_sd[weight_key]):
            skipped += 1
            continue

        rank = down.shape[0]
        alpha_t = lora_sd.get(base + ".alpha")
        scale = (alpha_t.item() / rank) if alpha_t is not None else 1.0

        target = merged_sd[weight_key]
        delta = up.float() @ down.float()
        if delta.shape != target.shape:
            skipped += 1
            continue
        merged_sd[weight_key] = (target.float() + strength * scale * delta).to(target.dtype).contiguous()
        applied += 1

    if skipped_te > 0:
        log.append(f"    Note: {skipped_te} text-encoder LoRA layers were skipped (only diffusion model layers are merged).")
    return applied, skipped


class StarModelAIOMerge:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = [NONE] + sorted(folder_paths.get_filename_list("loras"))
        weight = {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
        strength = {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}
        return {
            "required": {
                "model_1": ("MODEL",),
                "weight_1": ("FLOAT", weight),
                "save_name": ("STRING", {"default": "Merge_"}),
            },
            "optional": {
                "model_2": ("MODEL",),
                "weight_2": ("FLOAT", weight),
                "model_3": ("MODEL",),
                "weight_3": ("FLOAT", weight),
                "model_4": ("MODEL",),
                "weight_4": ("FLOAT", weight),
                "lora_1": (lora_list, {}),
                "lora_strength_1": ("FLOAT", strength),
                "lora_2": (lora_list, {}),
                "lora_strength_2": ("FLOAT", strength),
                "text_encoder": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = "Merges 1-4 connected diffusion models of the same architecture (Qwen Image, Flux, SD3.5, WAN, Z-Image, ...) with individual weights, optionally bakes in LoRAs and bundles text encoder + VAE into one AIO checkpoint."

    def merge(self, model_1, weight_1, save_name,
              model_2=None, weight_2=1.0, model_3=None, weight_3=1.0, model_4=None, weight_4=1.0,
              lora_1=NONE, lora_strength_1=1.0, lora_2=NONE, lora_strength_2=1.0,
              text_encoder=None, vae=None):

        log = []
        log.append("=" * 60)
        log.append("Star Model And AIO Merge - Starting")
        log.append("=" * 60)

        # ---------------- merge diffusion models ----------------
        log.append("Reading diffusion weights of model #1 (base)...")
        merged_sd = _model_sd(model_1)

        others = []
        for i, (m, w) in enumerate(((model_2, weight_2), (model_3, weight_3), (model_4, weight_4)), 2):
            if m is None or w == 0.0:
                continue
            log.append(f"Reading diffusion weights of model #{i} (weight {w:.2f})...")
            others.append((f"model #{i}", _model_sd(m), w))

        for label, w in [(("model #1"), weight_1)] + [(n, w) for n, _, w in others]:
            log.append(f"Model  : {label}  (weight {w:.2f})")

        keys = list(merged_sd.keys())
        total = len(keys)
        log.append(f"Merging {total} tensors in 32-bit precision...")
        t_start = time.time()
        missing_in = {}
        shared_in = {}
        quant_keys = 0
        merged_keys = 0

        for idx, k in enumerate(keys, 1):
            base_t = merged_sd[k]
            if any(k.endswith(sfx) for sfx in QUANT_HELPER_SUFFIXES):
                # quantization helper values (scales etc.) - kept for now, dropped later
                # if the base model turned out to be quantized
                continue

            rb = _real_weight(base_t)
            if rb is None:
                continue
            if _is_quantized(base_t):
                quant_keys += 1

            acc = rb * weight_1
            w_sum = weight_1
            for name, sd, w in others:
                t = sd.get(k)
                rt = _real_weight(t) if t is not None else None
                if rt is None or rt.shape != rb.shape:
                    if t is None:
                        missing_in[name] = missing_in.get(name, 0) + 1
                    continue
                shared_in[name] = shared_in.get(name, 0) + 1
                acc += rt * w
                w_sum += w
                del rt

            out_dtype = torch.bfloat16 if _is_quantized(base_t) else base_t.dtype
            merged_sd[k] = (acc / w_sum).to(out_dtype).contiguous()
            del acc, rb
            merged_keys += 1

            if idx % 250 == 0 or idx == total:
                print(f"  Tensor {idx}/{total}...")

        # drop helper keys of quantized base models from the merged file -
        # the merged weights are real values again, stale scales would corrupt them
        if quant_keys > 0:
            for k in list(merged_sd.keys()):
                if any(k.endswith(sfx) for sfx in QUANT_HELPER_SUFFIXES):
                    del merged_sd[k]

        warnings = []
        for name, sd, _ in others:
            shared = shared_in.get(name, 0)
            missing = missing_in.get(name, 0)
            if shared == 0:
                warnings.append(f"{name} shares 0 mergeable tensors with model #1 - different architecture (e.g. not a Qwen Image 2.1 model)? The model was IGNORED.")
            elif missing > max(50, 0.1 * total):
                warnings.append(f"{name}: {missing} of {total} tensors not present/matching - check that the model really is the same architecture as model #1.")
        if quant_keys > 0:
            warnings.append(f"{quant_keys} tensors were quantized ({'int8/fp8'} scaled formats). They were dequantized before merging and the result is saved as BF16 - the merged file is larger than the inputs.")
        for msg in warnings:
            log.append(f"  WARNING: {msg}")
            print(f"  WARNING: {msg}")
        log.append(f"  Merged {merged_keys} tensors in {time.time() - t_start:.1f}s")
        del others
        gc.collect()

        # ---------------- optional LoRA bake-in ----------------
        for lora_name, lora_strength in ((lora_1, lora_strength_1), (lora_2, lora_strength_2)):
            if lora_name == NONE or lora_strength == 0.0:
                continue
            try:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            except Exception as e:
                return (f"LoRA not found: {e}",)
            log.append("")
            log.append(f"Applying LoRA : {lora_name}  (strength {lora_strength:.2f})")
            applied, skipped = _apply_lora(merged_sd, lora_path, lora_strength, log)
            log.append(f"  {applied} layers patched, {skipped} skipped")
            gc.collect()

        # ---------------- optional AIO bundle ----------------
        aio = text_encoder is not None or vae is not None

        clip_sd = None
        if text_encoder is not None:
            log.append("")
            log.append("Adding connected text encoder (CLIP)...")
            try:
                clip_sd = text_encoder.state_dict_for_saving()
            except Exception as e:
                return (f"Failed to read text encoder state dict: {e}",)
            gc.collect()

        vae_sd = None
        if vae is not None:
            log.append("Adding connected VAE...")
            try:
                vae_sd = vae.get_sd()
            except Exception as e:
                return (f"Failed to read VAE state dict: {e}",)
            gc.collect()

        # ---------------- save ----------------
        safe_name = "".join(c for c in save_name.strip() if c not in '\\/:*?"<>|')
        if not safe_name:
            safe_name = "Merge_"
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        out_dir = os.path.join(folder_paths.get_output_directory(), "checkpoints" if aio else "models")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{safe_name}{timestamp}.safetensors")

        log.append("")
        if aio:
            log.append("Saving AIO checkpoint (model + text encoder + VAE)...")
            final_sd = {}
            for k, v in merged_sd.items():
                final_sd["model.diffusion_model." + k] = v if v.is_contiguous() else v.contiguous()
            if clip_sd is not None:
                for k, v in clip_sd.items():
                    final_sd["text_encoders." + k] = v
            if vae_sd is not None:
                for k, v in vae_sd.items():
                    final_sd["vae." + k] = v
        else:
            log.append("Saving merged diffusion model...")
            final_sd = {}
            for k, v in merged_sd.items():
                final_sd[k] = v if v.is_contiguous() else v.contiguous()

        model_entries = [{"slot": "model_1", "weight": weight_1}]
        for i, (m, w) in enumerate(((model_2, weight_2), (model_3, weight_3), (model_4, weight_4)), 2):
            if m is not None and w != 0.0:
                model_entries.append({"slot": f"model_{i}", "weight": w})
        metadata = {"merge_info": json.dumps({
            "type": "aio_checkpoint" if aio else "merged_diffusion_model",
            "models": model_entries,
            "loras": [{"file": n, "strength": s} for n, s in ((lora_1, lora_strength_1), (lora_2, lora_strength_2)) if n != NONE and s != 0.0],
            "text_encoder": text_encoder is not None,
            "vae": vae is not None,
        })}

        try:
            comfy.utils.save_torch_file(final_sd, out_path, metadata=metadata)
        except Exception as e:
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return (f"Failed to save checkpoint: {e}",)

        size = os.path.getsize(out_path)
        log.append("")
        log.append("=" * 60)
        log.append("Done!")
        log.append("=" * 60)
        log.insert(3, f"Saved to: {out_path}")

        info = (
            f"Merge finished!\n"
            f"Saved to: {out_path}\n"
            f"File size: {_fmt_size(size)}\n"
            f"Type: {'AIO checkpoint (model + TE + VAE)' if aio else 'merged diffusion model'}\n"
            f"Component tensors merged: {merged_keys}\n"
            f"Models: " + ", ".join(f"{e['slot']} x{e['weight']:.2f}" for e in model_entries) + "\n"
            f"LoRAs baked in: " + (", ".join(f"{n} x{s:.2f}" for n, s in ((lora_1, lora_strength_1), (lora_2, lora_strength_2)) if n != NONE) or "none") + "\n"
            + ("".join(f"\nWARNING: {w}" for w in warnings) if warnings else "")
            + "\nRestart ComfyUI or press 'r' in the loaders to use the file from a dropdown, or load it right away from the output folder."
        )

        print("\n".join(log))
        return (info,)


NODE_CLASS_MAPPINGS = {
    "StarModelAIOMerge": StarModelAIOMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarModelAIOMerge": "⭐ Star Model And AIO Merge",
}
