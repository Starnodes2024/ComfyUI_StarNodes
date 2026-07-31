import os
import torch
import folder_paths
import comfy.utils
import comfy.sd


def _get_lora_list():
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []
    return ["None"] + loras


LORA_LIST = _get_lora_list()


class DynamicLoraInputs(dict):
    """Flexible parameter definition for dynamic LoRA slots (model + clip)."""

    def __getitem__(self, key):
        # LoRA name
        if key.startswith("lora") and key.endswith("_name"):
            return (LORA_LIST, {"default": "None"})
        # Model strength
        if key.startswith("strength") and key.endswith("_model"):
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        # CLIP strength
        if key.startswith("strength") and key.endswith("_clip"):
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        # Fallback
        return ("FLOAT", {"default": 0.0})

    def __contains__(self, key):
        return True


class DynamicLoraInputsModelOnly(dict):
    """Flexible parameter definition for dynamic LoRA slots (model only)."""

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


class StarDynamicLora:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Sampler"

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        base_optional = {
            "lora1_name": (LORA_LIST, {"default": "None", "tooltip": "First LoRA to apply. This slot is always present."}),
            "strength1_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "strength1_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply LoRAs to."}),
                "clip": ("CLIP", {"tooltip": "CLIP to apply LoRAs to."}),
            },
            "optional": DynamicLoraInputs(base_optional),
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    DESCRIPTION = "Dynamically apply any number of LoRAs with model and CLIP strengths. First LoRA slot is fixed; more can be added from the UI."

    def _load_lora(self, name):
        if name == "None":
            return None
        if name in self._lora_cache:
            return self._lora_cache[name]
        lora_path = folder_paths.get_full_path_or_raise("loras", name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._lora_cache[name] = lora
        return lora

    def apply_loras(self, model, clip, **kwargs):
        model_out = model
        clip_out = clip

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

        for idx in lora_indices:
            name_key = f"lora{idx}_name"
            sm_key = f"strength{idx}_model"
            sc_key = f"strength{idx}_clip"

            lora_name = kwargs.get(name_key, "None")
            strength_model = kwargs.get(sm_key, 1.0)
            strength_clip = kwargs.get(sc_key, 1.0)

            if lora_name == "None" or strength_model == 0:
                continue

            lora = self._load_lora(lora_name)
            if lora is None:
                continue

            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out,
                clip_out,
                lora,
                float(strength_model),
                float(strength_clip) if clip_out is not None else 0.0,
            )

        return (model_out, clip_out)


class StarDynamicLoraModelOnly:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Sampler"

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        base_optional = {
            "lora1_name": (LORA_LIST, {"default": "None", "tooltip": "First LoRA to apply. This slot is always present."}),
            "strength1_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply LoRAs to."}),
            },
            "optional": DynamicLoraInputsModelOnly(base_optional),
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_loras"
    DESCRIPTION = "Dynamically apply any number of LoRAs to the model only. First LoRA slot is fixed; more can be added from the UI."

    def _load_lora(self, name):
        if name == "None":
            return None
        if name in self._lora_cache:
            return self._lora_cache[name]
        lora_path = folder_paths.get_full_path_or_raise("loras", name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._lora_cache[name] = lora
        return lora

    def apply_loras(self, model, **kwargs):
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

        for idx in lora_indices:
            name_key = f"lora{idx}_name"
            sm_key = f"strength{idx}_model"

            lora_name = kwargs.get(name_key, "None")
            strength_model = kwargs.get(sm_key, 1.0)

            if lora_name == "None" or strength_model == 0:
                continue

            lora = self._load_lora(lora_name)
            if lora is None:
                continue

            model_out, _ = comfy.sd.load_lora_for_models(
                model_out,
                None,
                lora,
                float(strength_model),
                0.0,
            )

        return (model_out,)


NODE_CLASS_MAPPINGS = {
    "StarDynamicLora": StarDynamicLora,
    "StarDynamicLoraModelOnly": StarDynamicLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarDynamicLora": "⭐ Star Dynamic LoRA",
    "StarDynamicLoraModelOnly": "⭐ Star Dynamic LoRA (Model Only)",
}
