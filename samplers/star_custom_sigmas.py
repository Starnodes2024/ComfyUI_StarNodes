import logging
import re

import torch


# Baked flow-match schedules. "Custom" uses the sigmas text field.
SIGMA_PRESETS = {
    "8 Steps": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0",
    "10 Steps": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.870114, 0.845148, 0.704534, 0.559551, 0.414568, 0.0",
    "12 Steps": "1.0, 0.985635, 0.970362, 0.95418, 0.935811, 0.916111, 0.89508, 0.861792, 0.798277, 0.704534, 0.511223, 0.276379, 0.0",
    "16 Steps": "1.0, 0.989227, 0.978453, 0.966317, 0.954180, 0.940403, 0.926626, 0.910853, 0.895080, 0.870114, 0.845148, 0.774841, 0.704534, 0.559551, 0.414568, 0.207284, 0.0",
}
CUSTOM = "Custom"


class StarCustomSigmas:
    """
    ⭐ Star Custom Sigmas

    Turns a comma-separated sigma list into a real SIGMAS output: the
    sampling run uses exactly these sigma values - N values mean N - 1
    sampling steps and the list usually ends at 0.0 for a full denoise.

    Pick a preset (8/10/12/16 steps) or choose "Custom" to use the text field
    as-is. Connect to the "options" input of ⭐ StarSampler (Unified) to
    override its steps/scheduler widgets, or to any vanilla ComfyUI SIGMAS
    input (e.g. SamplerCustomAdvanced).
    """
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"

    DEFAULT_SIGMAS = SIGMA_PRESETS["10 Steps"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(SIGMA_PRESETS.keys()) + [CUSTOM],
                    {
                        "default": "10 Steps",
                        "tooltip": "Baked sigma schedule: 8/10/12/16 steps, or 'Custom' to use the sigmas text field as-is.",
                    },
                ),
                "sigmas": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": cls.DEFAULT_SIGMAS,
                        "tooltip": "Comma-separated sigma values, high to low. Steps = values - 1 - only used when the preset is 'Custom' (editing this field switches the preset to 'Custom'). End with 0.0 for a full denoise.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Sampler"
    DESCRIPTION = "Exact custom sigma schedule for sampling (8/10/12/16-step presets or a free-form list). Connect to the options input of ⭐ StarSampler (Unified) or any SIGMAS input - N values mean N-1 sampling steps."

    def create(self, preset, sigmas):
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.*\d+)", SIGMA_PRESETS.get(preset, sigmas))]
        if len(values) < 2:
            logging.warning("StarCustomSigmas: need at least 2 sigma values, falling back to '1.0, 0.0'")
            values = [1.0, 0.0]
        return (torch.FloatTensor(values),)


NODE_CLASS_MAPPINGS = {
    "StarCustomSigmas": StarCustomSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarCustomSigmas": "⭐ Star Custom Sigmas",
}
