"""
Star Minimax H3 LoRA tools (merged from the standalone Star_M3H_Merger pack).

Nodes
-----
StarMinimaxH3LoraLoader   Load a LoRA file into the MINIMAX_H3_LORA pipeline type
StarMinimaxH3LoraMerge    Merge two LoRAs into one (SVD re-composition, output rank)
StarMinimaxH3LoraSaver    Save a merged LoRA to disk (default: loras folder)
"""

import os

import torch
from safetensors.torch import load_file as st_load_file

import folder_paths

from . import minimax_h3_merge_utils as U

CATEGORY = "⭐StarNodes/Video"
LORA_TYPE = "MINIMAX_H3_LORA"
RANK_CHOICES = [str(r) for r in range(8, 129, 8)]  # 8, 16, ..., 128


def _lora_list():
    return sorted(folder_paths.get_filename_list("loras"))


# ---------------------------------------------------------------------------
# 1. LoRA loader
# ---------------------------------------------------------------------------
class StarMinimaxH3LoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name": (_lora_list(),),
        }}

    RETURN_TYPES = (LORA_TYPE, "STRING")
    RETURN_NAMES = ("minimax_h3_lora", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a MiniMax-H3 LoRA (.safetensors) for the Star Minimax H3 LoRA merge pipeline."

    def load(self, lora_name):
        path = folder_paths.get_full_path("loras", lora_name)
        sd = st_load_file(path, device="cpu")
        pairs, dangling, others = U.parse_lora(sd)
        report = (f"Loaded '{lora_name}': {len(pairs)} LoRA pairs, "
                  f"{len(others)} extra tensors"
                  + (f", {len(dangling)} incomplete pairs (ignored)" if dangling else ""))
        U.log(report)
        return ({"sd": sd, "name": lora_name}, report)


# ---------------------------------------------------------------------------
# 2. LoRA merge
# ---------------------------------------------------------------------------
class StarMinimaxH3LoraMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_a": (LORA_TYPE,),
            "lora_b": (LORA_TYPE,),
            "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "0 = only LoRA A, 0.5 = 50/50, 1 = only LoRA B"}),
            "output_rank": (RANK_CHOICES, {"default": "32",
                                           "tooltip": "Rank of the merged LoRA. If the combined rank of A+B is smaller, that is used instead (nothing lost)."}),
            "output_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
        }}

    RETURN_TYPES = (LORA_TYPE, "STRING")
    RETURN_NAMES = ("minimax_h3_lora", "report")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Merge two H3 LoRAs: deltas are blended in weight space and "
                   "re-composed to up/down factors via SVD, so the result stays a "
                   "normal LoRA usable everywhere.")

    def merge(self, lora_a, lora_b, weight, output_rank, output_dtype):
        sd_a, sd_b = lora_a["sd"], lora_b["sd"]
        pa, da, oa = U.parse_lora(sd_a)
        pb, db, ob = U.parse_lora(sd_b)
        out_dt = U.OUT_DTYPES[output_dtype]
        rank_req = int(output_rank)
        wa, wb = 1.0 - weight, weight
        out, notes = {}, []

        for base in sorted(set(pa) | set(pb)):
            in_a, in_b = base in pa, base in pb
            if in_a and in_b:
                dw = U.lora_delta(pa[base]).mul_(wa).add_(U.lora_delta(pb[base]), alpha=wb)
                combined = pa[base]["up"].shape[1] + pb[base]["up"].shape[1]
                rank = min(rank_req, combined)
                up, down = U.svd_recompose(dw, rank)
                out[base + ".lora_up.weight"] = up.to(out_dt)
                out[base + ".lora_down.weight"] = down.to(out_dt)
                out[base + ".alpha"] = torch.tensor(float(up.shape[1]))
                del dw
            elif in_a:
                out[base + ".lora_up.weight"] = (pa[base]["up"].to(torch.float32) * wa).to(out_dt)
                out[base + ".lora_down.weight"] = pa[base]["down"].to(out_dt)
                if "alpha" in pa[base]:
                    out[base + ".alpha"] = pa[base]["alpha"]
            else:
                out[base + ".lora_up.weight"] = (pb[base]["up"].to(torch.float32) * wb).to(out_dt)
                out[base + ".lora_down.weight"] = pb[base]["down"].to(out_dt)
                if "alpha" in pb[base]:
                    out[base + ".alpha"] = pb[base]["alpha"]

        for k in sorted(set(oa) | set(ob)):
            if k in oa and k in ob and oa[k].shape == ob[k].shape \
                    and oa[k].is_floating_point():
                out[k] = (oa[k].to(torch.float32) * wa + ob[k].to(torch.float32) * wb).to(out_dt)
                notes.append(f"blended extra tensor {k}")
            else:
                src = oa.get(k, ob.get(k))
                out[k] = src.to(out_dt) if src.is_floating_point() else src
                notes.append(f"kept extra tensor {k} from one side")

        n = sum(1 for k in out if k.endswith(".lora_up.weight"))
        report = (f"LoRA merge @ w={weight:.2f}, rank<={rank_req}: {n} pairs written "
                  f"({len(set(pa) & set(pb))} joint, "
                  f"{len(set(pa) ^ set(pb))} single-sided scaled), "
                  f"{len(oa) + len(ob)} extra tensors handled.")
        U.log(report)
        name = f"merged({lora_a.get('name', '?')},{lora_b.get('name', '?')})"
        return ({"sd": out, "name": name}, report)


# ---------------------------------------------------------------------------
# 3. LoRA saver
# ---------------------------------------------------------------------------
class StarMinimaxH3LoraSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "minimax_h3_lora": (LORA_TYPE,),
            "location": (["loras folder", "output folder", "custom path"],),
            "filename": ("STRING", {"default": "minimax_h3_lora_merged.safetensors"}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Save a merged MiniMax-H3 LoRA to disk (standard ComfyUI LoRA format)."

    def save(self, minimax_h3_lora, location, filename, overwrite):
        dest = U.resolve_save_path(location, filename)
        if os.path.exists(dest) and not overwrite:
            raise FileExistsError(f"{dest} already exists (enable overwrite).")
        sd = minimax_h3_lora["sd"]
        plan = [(k, v.dtype, tuple(v.shape)) for k, v in sorted(sd.items())]
        U.write_safetensors_stream(dest, plan, lambda k: sd[k],
                                   {"star_minimax_h3": "merged lora",
                                    "source": minimax_h3_lora.get("name", "")})
        if "loras" in dest:
            try:
                folder_paths.get_filename_list.cache_clear()
            except (AttributeError, TypeError):
                pass
        U.log(f"lora saved -> {dest}")
        return (dest,)


NODE_CLASS_MAPPINGS = {
    "StarMinimaxH3LoraLoader": StarMinimaxH3LoraLoader,
    "StarMinimaxH3LoraMerge": StarMinimaxH3LoraMerge,
    "StarMinimaxH3LoraSaver": StarMinimaxH3LoraSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxH3LoraLoader": "⭐ Star Minimax H3 LoRA Loader",
    "StarMinimaxH3LoraMerge": "⭐ Star Minimax H3 LoRA Merge",
    "StarMinimaxH3LoraSaver": "⭐ Star Minimax H3 LoRA Saver",
}
