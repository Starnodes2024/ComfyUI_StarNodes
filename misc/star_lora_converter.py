"""Star LoRA Converter

Converts an existing LoRA from models/loras: resize its rank with SVD
(lora_up @ lora_down -> truncated singular value decomposition), optionally
re-cast the weights to FP16 / FP8 (e4m3fn), and save the
result as a new .safetensors file - with a human-readable conversion report.

Supports the common LoRA key conventions:
    *.lora_down.weight / *.lora_up.weight / *.alpha   (Kohya / LyCORIS)
    *.lora_A.weight    / *.lora_B.weight  / *.alpha   (Diffusers / PEFT)

Linear (2D) and Conv2d (4D) modules are both handled - conv tensors are
flattened for the SVD and reshaped back afterwards. Every other tensor
(biases, norms, lokr/hada parts, embeddings, ...) is passed through
unchanged apart from the optional precision cast.
"""

import logging
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.utils
import folder_paths

_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")
_ALPHA_SUFFIX = ".alpha"

_RANK_OPTIONS = ["keep", "128", "64", "32", "16", "8", "4", "2", "1"]
_PRECISIONS = ["keep", "FP16", "FP8 (e4m3fn)"]
_PREC_TAGS = {"keep": "", "FP16": "_fp16", "FP8 (e4m3fn)": "_fp8"}


def _pair(key):
    """Return (base, role) when the key is a LoRA down/up weight."""
    for suf in _DOWN_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "down", suf
    for suf in _UP_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "up", suf
    return None


def _to_2d(t, role):
    """Flatten a lora weight to 2D for the SVD math."""
    if role == "down":
        return t.reshape(t.shape[0], -1)            # (r, in*k*k)
    return t.reshape(t.shape[0], -1).t()            # (out, r) -> keep (out,r)


class StarLoraConverter:
    """Resize the rank of an existing LoRA via SVD and optionally re-cast
    its weights (FP16 / FP8 e4m3fn), saving a new .safetensors file
    inside models/loras/<save_subfolder>."""

    BGCOLOR = "#4d1212"
    COLOR = "#1d124d"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "The LoRA to convert (from models/loras)."}),
                "target_rank": (_RANK_OPTIONS, {
                    "default": "32",
                    "tooltip": "New rank for the LoRA modules. 'keep' only "
                               "applies the precision cast, no SVD."}),
                "precision": (_PRECISIONS, {
                    "default": "FP16",
                    "tooltip": "Weight precision of the saved file. Alpha, "
                               "norm, bias and other non-LoRA tensors keep "
                               "a loader-safe float dtype."}),
                "output_filename_prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Base name of the output file. Empty = the "
                               "original filename. _r<rank>/_fp16/... tags "
                               "are appended automatically."}),
                "save_subfolder": ("STRING", {
                    "default": "resized",
                    "tooltip": "Subfolder inside models/loras for the "
                               "converted file."}),
                "svd_device": (["cuda", "cpu"], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "tooltip": "Where the SVD math runs (always in fp32). "
                               "cpu is safest for big text encoders."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("conversion_report",)
    FUNCTION = "convert"
    DESCRIPTION = ("Resize a LoRA's rank with SVD and re-cast its weights "
                   "(FP16/FP8), saving a new file into models/loras. "
                   "Returns a conversion report.")

    def convert(self, lora_name, target_rank, precision,
                output_filename_prefix, save_subfolder, svd_device):
        started = time.time()
        src = folder_paths.get_full_path_or_raise("loras", lora_name)
        if svd_device == "cuda" and not torch.cuda.is_available():
            logging.warning("[Star LoRA Converter] no CUDA - SVD on cpu")
            svd_device = "cpu"
        device = torch.device(svd_device)
        new_rank = None if target_rank == "keep" else int(target_rank)
        if precision == "FP8 (e4m3fn)" and not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError(
                "Star LoRA Converter: FP8 (e4m3fn) is not supported by this "
                "PyTorch build - pick FP16 or keep.")

        state, meta = {}, None
        with safe_open(src, framework="pt") as f:
            meta = f.metadata() or {}
            for key in f.keys():
                state[key] = f.get_tensor(key)

        # group the lora pairs by module base key
        pairs = {}
        solo, passthrough = [], []
        for key in state:
            hit = _pair(key)
            if hit is None:
                passthrough.append(key)
            else:
                pairs.setdefault(hit[0], {})[hit[1]] = (key, hit[2])

        out, ranks_seen = {}, set()
        n_resized = n_kept = 0
        for key in passthrough:                      # non-pair tensors incl.
            out[key] = state[key]                    # unmatched down/up keys

        pbar = (comfy.utils.ProgressBar(len(pairs))
                if hasattr(comfy.utils, "ProgressBar") else None)
        logging.info("[Star LoRA Converter] %s: %d LoRA module(s), "
                     "target rank %s, %s on %s",
                     lora_name, len(pairs), target_rank, precision, svd_device)
        log_every = max(1, len(pairs) // 20)         # ~5 % steps in console

        for i, (base, roles) in enumerate(pairs.items()):
            if pbar is not None:
                pbar.update_absolute(i + 1)
            if (i + 1) % log_every == 0 or i + 1 == len(pairs):
                logging.info("[Star LoRA Converter] %d/%d modules...",
                             i + 1, len(pairs))
            if not ("down" in roles and "up" in roles):
                solo.append(base)
                for k, _s in roles.values():
                    out[k] = state[k]
                continue
            down_key, down_suf = roles["down"]
            up_key, _ = roles["up"]
            alpha_key = base + _ALPHA_SUFFIX
            down, up = state[down_key], state[up_key]
            cur_rank = down.shape[0]
            ranks_seen.add(cur_rank)
            alpha = state.get(alpha_key)

            if new_rank is None or cur_rank <= new_rank:
                # nothing to resize - keep the module as it is
                out[down_key], out[up_key] = down, up
                if alpha_key in state:
                    out[alpha_key] = state[alpha_key]
                n_kept += 1
                continue

            try:
                # delta = up @ down has rank <= cur_rank, so the exact
                # truncated SVD comes from the tiny (r x r) core after a QR
                # of both factors - no big dense delta, much faster.
                up2d = up.to(device, torch.float32).reshape(up.shape[0], -1)
                down2d = down.to(device, torch.float32).reshape(down.shape[0], -1)
                Qu, Ru = torch.linalg.qr(up2d)       # (out,k) (k,k)
                Qd, Rd = torch.linalg.qr(down2d.t()) # (in,k)  (k,k)
                core = Ru @ Rd.t()
                if alpha is not None:
                    core = core * (float(alpha.reshape(-1)[0].item()) / cur_rank)
                Um, S, VmT = torch.linalg.svd(core, full_matrices=False)
                r = min(new_rank, S.shape[0])
                sqrt_S = torch.sqrt(S[:r])
                up_new = ((Qu @ Um[:, :r]) * sqrt_S).reshape(
                    up.shape[0], r, *up.shape[2:])
                down_new = ((VmT[:r] @ Qd.t()) * sqrt_S.unsqueeze(1)).reshape(
                    r, *down.shape[1:])
                out[up_key] = up_new.cpu().contiguous()
                out[down_key] = down_new.cpu().contiguous()
                out[alpha_key] = torch.tensor(float(r), dtype=torch.float32)
                del up2d, down2d, Qu, Ru, Qd, Rd, core, Um, S, VmT, up_new, down_new
                n_resized += 1
            except Exception as e:
                logging.warning("[Star LoRA Converter] skip %s: %s", base, e)
                out[down_key], out[up_key] = down, up
                if alpha_key in state:
                    out[alpha_key] = state[alpha_key]
                n_kept += 1

        # precision pass - weight tensors only; alpha/scalars/bias/norm keep
        # a loader-safe dtype
        weight_keys = {k for roles in pairs.values() for k, _s in roles.values()}
        final = {}
        for key, t in out.items():
            if key not in weight_keys or precision == "keep":
                final[key] = t if t.is_contiguous() else t.contiguous()
            elif precision == "FP16":
                final[key] = t.to(torch.float16)
            elif precision == "FP8 (e4m3fn)":
                final[key] = t.to(torch.float8_e4m3fn)

        # metadata: preserve, refresh the dim/alpha entries
        meta = dict(meta)
        meta["resized_by_node"] = "Star LoRA Converter"
        if new_rank is not None:
            meta["ss_network_dim"] = str(new_rank)
            meta["ss_network_alpha"] = str(new_rank)

        stem = (output_filename_prefix.strip()
                or os.path.splitext(os.path.basename(lora_name))[0])
        tag = (f"_r{new_rank}" if new_rank is not None else "") \
            + _PREC_TAGS[precision]
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        out_dir = os.path.join(lora_dir, save_subfolder.strip() or "resized")
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, f"{stem}{tag}.safetensors")
        n = 2                                       # never overwrite
        while os.path.exists(dst):
            dst = os.path.join(out_dir, f"{stem}{tag}_v{n}.safetensors")
            n += 1
        save_file(final, dst, metadata=meta)

        del state, out, final
        if svd_device == "cuda":
            torch.cuda.empty_cache()

        src_mb = os.path.getsize(src) / 1e6
        dst_mb = os.path.getsize(dst) / 1e6
        ranks = " / ".join(str(r) for r in sorted(ranks_seen)) or "none"
        report = [
            "Star LoRA Converter - report",
            f"  source:        {src}",
            f"  saved:         {dst}",
            f"  ranks found:   {ranks}  ->  target: {target_rank}",
            f"  modules:       {n_resized} resized, {n_kept} kept "
            f"(rank <= target or 'keep'), {len(passthrough) + len(solo)} "
            f"tensor(s) passed through",
            f"  precision:     {precision}",
            f"  size:          {src_mb:.1f} MB -> {dst_mb:.1f} MB "
            f"({100 - dst_mb / src_mb * 100:.1f}% smaller)",
            f"  time:          {time.time() - started:.1f} s",
        ]
        logging.info("[Star LoRA Converter] %s -> %s", lora_name, dst)
        return ("\n".join(report),)


NODE_CLASS_MAPPINGS = {
    "StarLoraConverter": StarLoraConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLoraConverter": "⭐ Star LoRA Converter",
}
