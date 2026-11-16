import os
import time

import torch
from safetensors.torch import save_file

try:
    from folder_paths import get_output_directory
except ImportError:
    def get_output_directory():
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        return out


def _state_dict_for_merge(patcher):
    """Raw tensors of the diffusion model, diffusion_model. prefix stripped."""
    sd = patcher.model.state_dict()
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model."):]
        out[k] = v.detach()
    return out


class StarModelMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                       "tooltip": "Weight for model_1. Weights are normalized, so only their ratio matters."}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                       "tooltip": "Weight for model_2. Weights are normalized, so only their ratio matters."}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                       "tooltip": "Weight for model_3 (only used when model_3 is connected)."}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                       "tooltip": "Weight for model_4 (only used when model_4 is connected)."}),
                "output_dtype": (["keep first model", "bf16", "fp16", "fp32"], {"default": "keep first model",
                                 "tooltip": "Dtype of the saved file. 'keep first model' uses the dtype of model_1."}),
                "save_folder": ("STRING", {"default": "merges", "multiline": False,
                                "tooltip": "Subfolder inside the ComfyUI output folder where the merged model is saved."}),
                "save_name": ("STRING", {"default": "merged_model", "multiline": False,
                              "tooltip": "File name of the merged model (.safetensors is appended if missing). Existing files are not overwritten - a counter is appended instead."}),
            },
            "optional": {
                "model_3": ("MODEL",),
                "model_4": ("MODEL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = ("Merges 2 to 4 diffusion models (weighted average) and saves the result "
                   "as a .safetensors file in a subfolder of the output folder. "
                   "Returns an info string with all merge details.")

    def merge(self, model_1, model_2, weight_1, weight_2, weight_3, weight_4,
              output_dtype, save_folder, save_name, model_3=None, model_4=None):
        started = time.time()

        models = [(model_1, weight_1), (model_2, weight_2),
                  (model_3, weight_3), (model_4, weight_4)]
        models = [(m, w) for m, w in models if m is not None]

        active = [(m, w) for m, w in models if w > 0.0]
        if len(active) < 2:
            return ("Nothing to merge: at least 2 connected models with a weight > 0 are needed.",)

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

        infos = []
        sds = []
        archs = set()
        for i, (m, w) in enumerate(models, 1):
            arch = type(m.model).__name__
            archs.add(arch)
            sd = _state_dict_for_merge(m)
            meta = sum(1 for t in sd.values() if t.device.type == "meta")
            if meta:
                return (f"Model {i} has unloaded (meta) weights - reload it with a normal model loader before merging.",)
            sds.append(sd)
            infos.append({"index": i, "arch": arch, "weight": w, "keys": len(sd)})
            print(f"⭐ StarNodes: Model Merge - model {i}: {arch}, {len(sd)} keys, weight {w}")

        # Active subset (weight > 0) with matching sd dicts
        active_pairs = [(sds[i], models[i][1]) for i in range(len(models)) if models[i][1] > 0.0]
        total_weight = sum(w for _, w in active_pairs)

        # Key order: first active model, then keys only found in later models
        keys = list(active_pairs[0][0].keys())
        seen = set(keys)
        for sd, _ in active_pairs[1:]:
            for k in sd:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        # Target dtype: first model's dtype, or the widget override
        ref_key = next((k for k in keys if active_pairs[0][0][k].dtype.is_floating_point), None)
        target_dtype = active_pairs[0][0][ref_key].dtype if ref_key is not None else torch.float32
        if output_dtype in dtype_map:
            target_dtype = dtype_map[output_dtype]

        merged = {}
        missing = []       # (key, model index) - model lacks this key
        mismatched = []    # (key, model index) - shape differs
        merged_keys = 0

        with torch.no_grad():
            for n, k in enumerate(keys, 1):
                acc = None
                ref_shape = None
                ref_dtype = None
                wsum = 0.0
                for mi, (sd, w) in enumerate(active_pairs, 1):
                    t = sd.get(k)
                    if t is None:
                        missing.append((k, mi))
                        continue
                    if acc is None and not t.dtype.is_floating_point:
                        # Non-floating tensors (buffers): take from the first model that has them
                        merged[k] = t.to("cpu")
                        wsum = -1.0
                        break
                    if ref_shape is None:
                        ref_shape = t.shape
                        ref_dtype = t.dtype
                    elif t.shape != ref_shape:
                        mismatched.append((k, mi))
                        continue
                    if acc is None:
                        acc = t.to("cpu", torch.float32) * w
                    else:
                        acc.add_(t.to("cpu", torch.float32), alpha=w)
                    wsum += w

                if wsum <= 0.0:
                    continue
                acc.div_(wsum)
                merged[k] = acc.to(ref_dtype if output_dtype == "keep first model" else target_dtype)
                merged_keys += 1
                if n % 200 == 0:
                    print(f"⭐ StarNodes: Model Merge - merged {n}/{len(keys)} keys...")

        # Output path
        folder = (save_folder or "merges").strip().strip("/\\")
        name = (save_name or "merged_model").strip()
        if not name.lower().endswith(".safetensors"):
            name += ".safetensors"
        out_dir = os.path.join(get_output_directory(), folder)
        os.makedirs(out_dir, exist_ok=True)

        dst = os.path.join(out_dir, name)
        counter = 1
        while os.path.exists(dst):
            counter += 1
            dst = os.path.join(out_dir, f"{name[:-len('.safetensors')]}_{counter:04d}.safetensors")

        save_file(merged, dst, metadata={"format": "pt"})
        del merged

        size_gb = os.path.getsize(dst) / (1024 ** 3)
        duration = time.time() - started
        print(f"⭐ StarNodes: Model Merge - saved {dst} ({size_gb:.2f} GB) in {duration:.1f}s")

        lines = [
            "Star Model Merge - info",
            "",
            "Models:",
        ]
        for info in infos:
            engaged = "yes" if info["weight"] > 0 else "no (weight 0 - skipped)"
            lines.append(f"  model_{info['index']}: {info['arch']}, keys: {info['keys']}, "
                         f"weight: {info['weight']} (merged: {engaged})")
        lines += [
            "",
            f"Architecture match: {'yes' if len(archs) == 1 else 'NO - models differ: ' + ', '.join(sorted(archs))}",
            f"Weight normalization: sum of active weights = {total_weight:.3f} (each tensor: weight / sum)",
            f"Merged tensors: {merged_keys} / {len(keys)} total keys",
            f"Keys missing in some models: {len(missing)}",
            f"Shape mismatches (skipped): {len(mismatched)}",
            f"Output dtype: {target_dtype} ({output_dtype})",
            "",
            f"Saved to: {dst}",
            f"File size: {size_gb:.2f} GB",
            f"Duration: {duration:.1f}s",
        ]
        if mismatched:
            lines += ["", "First mismatched keys:"]
            lines += [f"  {k} (model {mi})" for k, mi in mismatched[:10]]

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "StarModelMerge": StarModelMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarModelMerge": "⭐ Star Model Merge",
}
