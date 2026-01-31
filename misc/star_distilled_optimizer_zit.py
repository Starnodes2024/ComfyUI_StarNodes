import math

import comfy.samplers


class StarDistilledOptimizerZIT:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable the ZIT two-pass optimizer. When disabled, StarSampler ignores this options input.",
                    },
                ),
                "start_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "euler",
                        "tooltip": "Sampler used for the initial ZIT pass.",
                    },
                ),
                "refine_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep" if "res_multistep" in comfy.samplers.KSampler.SAMPLERS else "euler",
                        "tooltip": "Sampler used for the refinement pass (useful if res_multistep is unavailable).",
                    },
                ),
                "start_steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Steps for the initial ZIT pass.",
                    },
                ),
                "refine_steps": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Steps for the refinement pass (0 disables the refine pass).",
                    },
                ),
                "start_denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise used for the initial ZIT pass (typically 1.0).",
                    },
                ),
                "refine_denoise": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoise used for the refinement pass.",
                    },
                ),
                "patch_shift": (
                    "FLOAT",
                    {
                        "default": 2.55,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "ZIT model sampling shift parameter (ModelSamplingZImage-style patch).",
                    },
                ),
                "patch_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "ZIT model sampling multiplier parameter (ModelSamplingZImage-style patch).",
                    },
                ),
                "noise_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied to the generated noise latent.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Sampler"

    def create(self, enable, start_sampler, refine_sampler, start_steps, refine_steps, start_denoise, refine_denoise, patch_shift, patch_multiplier, noise_multiplier):
        payload = {
            "starnodes_type": "ZIT",
            "enabled": bool(enable),
            "start_sampler": start_sampler,
            "refine_sampler": refine_sampler,
            "start_steps": int(start_steps),
            "refine_steps": int(refine_steps),
            "start_denoise": float(start_denoise),
            "refine_denoise": float(refine_denoise),
            "patch_shift": float(patch_shift),
            "patch_multiplier": float(patch_multiplier),
            "noise_multiplier": float(noise_multiplier),
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "StarDistilledOptimizerZIT": StarDistilledOptimizerZIT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarDistilledOptimizerZIT": "⭐ Distilled Optimizer (QWEN/ZIT)",
}
