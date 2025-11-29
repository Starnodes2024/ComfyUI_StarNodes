import math
import torch

try:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers"])
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

# Default config for registering in ComfyUI scheduler list
# This matches the tuned defaults from the original FlowMatch node
# (good for Z-Image-Turbo / Flux-style models)
DEFAULT_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _flowmatch_scheduler_handler(model_sampling, steps):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(DEFAULT_CONFIG)
    scheduler.set_timesteps(steps, device=getattr(model_sampling, "device", "cpu"), mu=0.0)
    return scheduler.sigmas


# Register the scheduler in ComfyUI (if not already added)
_STAR_FM_SCHED_NAME = "FlowMatch Euler DiscScheduler"

if _STAR_FM_SCHED_NAME not in SCHEDULER_HANDLERS:
    handler = SchedulerHandler(handler=_flowmatch_scheduler_handler, use_ms=True)
    SCHEDULER_HANDLERS[_STAR_FM_SCHED_NAME] = handler
    if _STAR_FM_SCHED_NAME not in SCHEDULER_NAMES:
        SCHEDULER_NAMES.append(_STAR_FM_SCHED_NAME)


class StarFlowMatchOption:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of diffusion steps. Z-Image-Turbo uses 9 steps by default (8 DiT forwards).",
                }),
                "base_image_seq_len": ("INT", {
                    "default": 256,
                    "tooltip": "Base sequence length for dynamic shifting. Should match model's training resolution (e.g., 256 for 512x512).",
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5,
                    "tooltip": "Stabilizes generation. Higher values = more consistent outputs. 0.5 is tuned for Turbo/Flux-style models.",
                }),
                "invert_sigmas": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Reverses the sigma schedule. Keep disabled unless experimenting.",
                }),
                "max_image_seq_len": ("INT", {
                    "default": 8192,
                    "tooltip": "Maximum sequence length for dynamic shifting (for large images).",
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15,
                    "tooltip": "Maximum variation allowed. 1.15 is tuned for Turbo/Flux-style models.",
                }),
                "num_train_timesteps": ("INT", {
                    "default": 1000,
                    "tooltip": "Timesteps the model was trained with (typically 1000).",
                }),
                "shift": ("FLOAT", {
                    "default": 3.0,
                    "tooltip": "Global timestep schedule shift. 3.0 is tuned for Turbo models.",
                }),
                "shift_terminal": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "End value for shifted schedule. 0.0 disables terminal shift.",
                }),
                "stochastic_sampling": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Adds controlled randomness to each step (ancestral-style).",
                }),
                "time_shift_type": (["exponential", "linear"], {
                    "default": "exponential",
                    "tooltip": "Method for resolution-dependent shifting.",
                }),
                "use_beta_sigmas": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Use beta-distributed sigmas (experimental).",
                }),
                "use_dynamic_shifting": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Auto-adjust timesteps based on image resolution.",
                }),
                "use_exponential_sigmas": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Use exponential sigma spacing.",
                }),
                "use_karras_sigmas": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Use Karras noise schedule for smoother results.",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Sampler"

    def create(
        self,
        steps,
        base_image_seq_len,
        base_shift,
        invert_sigmas,
        max_image_seq_len,
        max_shift,
        num_train_timesteps,
        shift,
        shift_terminal,
        stochastic_sampling,
        time_shift_type,
        use_beta_sigmas,
        use_dynamic_shifting,
        use_exponential_sigmas,
        use_karras_sigmas,
    ):
        config = {
            "base_image_seq_len": base_image_seq_len,
            "base_shift": base_shift,
            "invert_sigmas": invert_sigmas == "enable",
            "max_image_seq_len": max_image_seq_len,
            "max_shift": max_shift,
            "num_train_timesteps": num_train_timesteps,
            "shift": shift,
            "shift_terminal": shift_terminal if shift_terminal != 0.0 else None,
            "stochastic_sampling": stochastic_sampling == "enable",
            "time_shift_type": time_shift_type,
            "use_beta_sigmas": use_beta_sigmas == "enable",
            "use_dynamic_shifting": use_dynamic_shifting == "enable",
            "use_exponential_sigmas": use_exponential_sigmas == "enable",
            "use_karras_sigmas": use_karras_sigmas == "enable",
        }

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
        scheduler.set_timesteps(steps, device="cpu", mu=0.0)
        sigmas = scheduler.sigmas
        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "StarFlowMatchOption": StarFlowMatchOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFlowMatchOption": "⭐ Star FlowMatch Option",
}
