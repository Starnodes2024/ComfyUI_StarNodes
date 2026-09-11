import comfy.samplers


class StarSplitSamplerOption:
    """
    Star Split Sampler Option. Update 2.6.1

    Lets the user pick two different samplers and a step count for each.
    Connected to the "options" input of ⭐ StarSampler (Unified) or
    ⭐ Star SD Upscale Refiner Advanced, the sampling runs
    steps_1 + steps_2 steps in total: the first steps_1 steps are done with
    sampler_1, then - starting at step steps_1 + 1 - the run switches to
    sampler_2 for the remaining steps.

    Example: euler 6 steps + ddim 6 steps -> 12 steps total, at step 7 the
    sampler switches from euler to ddim.
    """
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
                        "tooltip": "Enable the split sampler option. When disabled, StarSampler / Star SD Upscale Refiner ignore this options input.",
                    },
                ),
                "sampler_1": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "euler",
                        "tooltip": "Sampler used for the first part of the run.",
                    },
                ),
                "steps_1": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of steps done with sampler_1.",
                    },
                ),
                "sampler_2": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "ddim" if "ddim" in comfy.samplers.KSampler.SAMPLERS else "euler",
                        "tooltip": "Sampler used for the second part of the run (starting at step steps_1 + 1).",
                    },
                ),
                "steps_2": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of steps done with sampler_2. Total steps = steps_1 + steps_2.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STARNODES_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Sampler"

    def create(self, enable, sampler_1, steps_1, sampler_2, steps_2):
        steps_1 = max(1, int(steps_1))
        steps_2 = max(1, int(steps_2))
        payload = {
            "starnodes_type": "SPLIT_SAMPLER",
            "enabled": bool(enable),
            "sampler_1": sampler_1,
            "steps_1": steps_1,
            "sampler_2": sampler_2,
            "steps_2": steps_2,
            "total_steps": steps_1 + steps_2,
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "StarSplitSamplerOption": StarSplitSamplerOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSplitSamplerOption": "⭐ Star Split Sampler Option",
}
