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
    """Flexible parameter definition for dynamic LoRA slots (model + clip).

    Each slot exposes three widgets:
      lora{N}_name      – combo (LoRA file or "None")
      strength{N}       – single float applied to both model and clip
      enabled{N}        – boolean toggle
    """

    def __getitem__(self, key):
        if key.startswith("lora") and key.endswith("_name"):
            return (LORA_LIST, {"default": "None"})
        if key.startswith("strength") and not key.endswith("_model") and not key.endswith("_clip"):
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        if key.startswith("enabled"):
            return ("BOOLEAN", {"default": True})
        return ("FLOAT", {"default": 0.0})

    def __contains__(self, key):
        return True


class DynamicLoraInputsModelOnly(dict):
    """Flexible parameter definition for dynamic LoRA slots (model only)."""

    def __getitem__(self, key):
        if key.startswith("lora") and key.endswith("_name"):
            return (LORA_LIST, {"default": "None"})
        if key.startswith("strength") and not key.endswith("_model") and not key.endswith("_clip"):
            return ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        if key.startswith("enabled"):
            return ("BOOLEAN", {"default": True})
        return ("FLOAT", {"default": 0.0})

    def __contains__(self, key):
        return True


def _parse_lora_indices(kwargs):
    indices = []
    for key in kwargs:
        if key.startswith("lora") and key.endswith("_name"):
            try:
                idx = int(key.replace("lora", "").replace("_name", ""))
                indices.append(idx)
            except ValueError:
                continue
    return sorted(set(indices))


class StarDynamicLora:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Sampler"

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        base_optional = {
            "clip": ("CLIP", {"tooltip": "CLIP to apply LoRAs to. Optional — if not connected, only the model is modified."}),
            "lora1_name": (LORA_LIST, {"default": "None", "tooltip": "First LoRA to apply. This slot is always present."}),
            "strength1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength applied to both model and CLIP."}),
            "enabled1": ("BOOLEAN", {"default": True, "tooltip": "Toggle this LoRA on or off."}),
        }
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply LoRAs to."}),
            },
            "optional": DynamicLoraInputs(base_optional),
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    DESCRIPTION = "Dynamically apply any number of LoRAs. Each slot has a single strength (used for both model and CLIP) and an on/off toggle. New slots appear automatically as you fill the last one. CLIP input is optional — leave it disconnected to apply LoRAs to the model only."

    def _load_lora(self, name):
        if name == "None":
            return None
        if name in self._lora_cache:
            return self._lora_cache[name]
        lora_path = folder_paths.get_full_path_or_raise("loras", name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._lora_cache[name] = lora
        return lora

    def apply_loras(self, model, clip=None, **kwargs):
        model_out = model
        clip_out = clip

        for idx in _parse_lora_indices(kwargs):
            lora_name = kwargs.get(f"lora{idx}_name", "None")
            strength = kwargs.get(f"strength{idx}", 1.0)
            enabled = kwargs.get(f"enabled{idx}", True)

            if not enabled or lora_name == "None" or strength == 0:
                continue

            lora = self._load_lora(lora_name)
            if lora is None:
                continue

            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out,
                clip_out,
                lora,
                float(strength),
                float(strength) if clip_out is not None else 0.0,
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
            "strength1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength applied to the model."}),
            "enabled1": ("BOOLEAN", {"default": True, "tooltip": "Toggle this LoRA on or off."}),
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
    DESCRIPTION = "Dynamically apply any number of LoRAs to the model only. Each slot has a single strength and an on/off toggle. New slots appear automatically as you fill the last one."

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

        for idx in _parse_lora_indices(kwargs):
            lora_name = kwargs.get(f"lora{idx}_name", "None")
            strength = kwargs.get(f"strength{idx}", 1.0)
            enabled = kwargs.get(f"enabled{idx}", True)

            if not enabled or lora_name == "None" or strength == 0:
                continue

            lora = self._load_lora(lora_name)
            if lora is None:
                continue

            model_out, _ = comfy.sd.load_lora_for_models(
                model_out,
                None,
                lora,
                float(strength),
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
