import os
import torch
import folder_paths
import comfy.utils
import comfy.sd
import copy


def _get_lora_list():
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []
    return ["None"] + loras


LORA_LIST = _get_lora_list()


class DynamicLoraNormalizerInputs(dict):
    """Flexible parameter definition for dynamic LoRA slots."""

    def __getitem__(self, key):
        # LoRA name
        if key.startswith("lora") and key.endswith("_name"):
            return (LORA_LIST, {"default": "None"})
        # Model strength
        if key.startswith("strength") and key.endswith("_model"):
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        # Fallback
        return ("FLOAT", {"default": 0.0})

    def __contains__(self, key):
        return True


class StarLoraWeightNormalizer:
    """
    Two modes for combining multiple LoRAs on distilled models:
    1. Normalize: Scale down weights to target total (e.g., 1.0)
    2. Blend: Merge LoRAs together at target weight
    """
    BGCOLOR = "#4d1212"
    COLOR = "#1d124d"
    CATEGORY = "⭐StarNodes/Sampler"

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        base_optional = {
            "lora1_name": (LORA_LIST, {"default": "None", "tooltip": "First LoRA to apply."}),
            "strength1_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply LoRAs to."}),
                "mode": (["normalize", "blend"], {
                    "default": "normalize",
                    "tooltip": "Normalize: scale weights down | Blend: merge LoRAs together"
                }),
                "target_weight": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Normalize: target sum of weights | Blend: strength of merged LoRA"
                }),
            },
            "optional": DynamicLoraNormalizerInputs(base_optional),
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply_loras"
    DESCRIPTION = """Two modes for multiple LoRAs on distilled models:

NORMALIZE MODE: Scales down individual LoRA weights so total = target
- LoRAs applied separately with reduced weights
- Maintains relative proportions
- Example: [0.8, 0.6, 0.4] → [0.44, 0.33, 0.22] (total 1.0)

BLEND MODE: Merges all LoRAs into one combined effect
- LoRAs blended together at equal ratios
- Applied as single combined LoRA at target weight
- Example: [0.8, 0.6, 0.4] → merged at 1.0 strength"""

    def _load_lora(self, name):
        if name == "None":
            return None
        if name in self._lora_cache:
            return self._lora_cache[name]
        lora_path = folder_paths.get_full_path_or_raise("loras", name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._lora_cache[name] = lora
        return lora

    def _blend_loras(self, lora_configs):
        """Blend multiple LoRAs into a single merged LoRA"""
        if not lora_configs:
            return None
        
        # Load all LoRAs
        loaded_loras = []
        for cfg in lora_configs:
            lora = self._load_lora(cfg["name"])
            if lora is not None:
                loaded_loras.append({
                    "lora": lora,
                    "strength": cfg["strength_model"],
                    "name": cfg["name"]
                })
        
        if not loaded_loras:
            return None
        
        # Calculate blend weights (normalize to sum to 1.0)
        total_strength = sum(abs(l["strength"]) for l in loaded_loras)
        blend_weights = [abs(l["strength"]) / total_strength for l in loaded_loras]
        
        # Create merged LoRA by weighted averaging
        merged_lora = {}
        
        for i, lora_data in enumerate(loaded_loras):
            lora = lora_data["lora"]
            weight = blend_weights[i]
            
            for key in lora.keys():
                if key not in merged_lora:
                    merged_lora[key] = lora[key].clone() * weight
                else:
                    merged_lora[key] = merged_lora[key] + (lora[key] * weight)
        
        return merged_lora

    def apply_loras(self, model, mode="normalize", target_weight=1.0, **kwargs):
        model_out = model

        # Find all lora slots
        lora_indices = []
        for key in kwargs.keys():
            if key.startswith("lora") and key.endswith("_name"):
                try:
                    idx = int(key.replace("lora", "").replace("_name", ""))
                    lora_indices.append(idx)
                except ValueError:
                    continue
        lora_indices = sorted(set(lora_indices))

        # Collect all LoRA configurations
        lora_configs = []
        for idx in lora_indices:
            name_key = f"lora{idx}_name"
            sm_key = f"strength{idx}_model"

            lora_name = kwargs.get(name_key, "None")
            strength_model = kwargs.get(sm_key, 1.0)

            if lora_name == "None" or strength_model == 0:
                continue

            lora_configs.append({
                "name": lora_name,
                "strength_model": float(strength_model),
                "index": idx
            })

        if not lora_configs:
            return (model_out, "No LoRAs applied")

        # MODE: NORMALIZE - Scale down weights
        if mode == "normalize":
            total_weight = sum(abs(cfg["strength_model"]) for cfg in lora_configs)
            norm_factor = target_weight / total_weight if total_weight > 0 else 1.0

            info_lines = [
                f"=== NORMALIZE MODE ===",
                f"Target Total Weight: {target_weight}",
                f"Original Total: {total_weight:.4f}",
                f"Normalization Factor: {norm_factor:.4f}",
                f"",
                f"Applied LoRAs (scaled down):",
            ]

            for cfg in lora_configs:
                lora = self._load_lora(cfg["name"])
                if lora is None:
                    continue

                normalized_strength = cfg["strength_model"] * norm_factor
                model_out, _ = comfy.sd.load_lora_for_models(
                    model_out, None, lora, normalized_strength, 0.0
                )

                info_lines.append(
                    f"  {cfg['name']}: {cfg['strength_model']:.3f} → {normalized_strength:.3f}"
                )

        # MODE: BLEND - Merge LoRAs together
        else:  # mode == "blend"
            total_strength = sum(abs(cfg["strength_model"]) for cfg in lora_configs)
            blend_weights = [abs(cfg["strength_model"]) / total_strength for cfg in lora_configs]

            info_lines = [
                f"=== BLEND MODE ===",
                f"Blending {len(lora_configs)} LoRAs together",
                f"Target Weight: {target_weight}",
                f"",
                f"Blend Ratios:",
            ]

            for i, cfg in enumerate(lora_configs):
                info_lines.append(
                    f"  {cfg['name']}: {blend_weights[i]*100:.1f}% (from {cfg['strength_model']:.3f})"
                )

            # Blend all LoRAs
            merged_lora = self._blend_loras(lora_configs)
            
            if merged_lora is not None:
                # Apply the merged LoRA at target weight
                model_out, _ = comfy.sd.load_lora_for_models(
                    model_out, None, merged_lora, target_weight, 0.0
                )
                info_lines.append(f"\nMerged LoRA applied at strength: {target_weight}")
            else:
                info_lines.append("\nError: Could not blend LoRAs")

        info_text = "\n".join(info_lines)
        print(info_text)

        return (model_out, info_text)


NODE_CLASS_MAPPINGS = {
    "StarLoraWeightNormalizer": StarLoraWeightNormalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLoraWeightNormalizer": "⭐ Star Dynamic LoRA Weight",
}
