import math

import torch
import comfy.model_sampling

# Custom scheduler entry prepended to the scheduler dropdown of the Star
# sampler nodes. Selecting it acts exactly as if a ⭐ Star LTX Scheduler
# options node were connected with its defaults.
STAR_LTX_SCHEDULER = "Star LTX4Qwen"

# Official Qwen-Image-2.1 scheduler config (FlowMatchEulerDiscreteScheduler,
# use_dynamic_shifting + time_shift_type="exponential").
QWEN2X_MAX_SHIFT = 0.9
QWEN2X_BASE_SHIFT = 0.5
QWEN2X_BASE_SEQ_LEN = 256
QWEN2X_MAX_SEQ_LEN = 8192
QWEN2X_SHIFT_TERMINAL = 0.02


def star_ltx_shift(tokens, max_shift=QWEN2X_MAX_SHIFT, base_shift=QWEN2X_BASE_SHIFT,
                   base_seq_len=QWEN2X_BASE_SEQ_LEN, max_seq_len=QWEN2X_MAX_SEQ_LEN):
    """Dynamic shift mu: linear interpolation over the image token count."""
    mm = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return tokens * mm + (base_shift - mm * base_seq_len)


class StarLTXModelSampling(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    """ModelSamplingFlux variant that makes the stock schedulers emit the
    Qwen Image 2.x LTX-style schedule: exponential dynamic shift with the
    tail stretched so the last scheduled sigma sits at `terminal` (Qwen's
    `shift_terminal`)."""
    _ltx_scale = 1.0

    def sigma(self, timestep):
        tt = torch.as_tensor(timestep, dtype=torch.float32)
        f = comfy.model_sampling.flux_time_shift(self.shift, 1.0, tt)
        out = torch.clamp(1.0 - (1.0 - f) / self._ltx_scale, min=0.0)
        return out if torch.is_tensor(timestep) else float(out)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return self.sigma(1.0 - percent)


def apply_star_ltx(model, tokens, steps, options=None):
    """Patch a model clone so the 'simple' scheduler produces the Qwen 2.x
    LTX-style dynamic-shift curve for `tokens` image tokens. Returns
    (model, shift); shift is None when the model is not Flux-style flow
    sampling and the patch was not applied."""
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        model_sampling = None
    if not isinstance(model_sampling, comfy.model_sampling.ModelSamplingFlux):
        return model, None

    opts = options or {}
    max_shift = float(opts.get("max_shift", QWEN2X_MAX_SHIFT))
    base_shift = float(opts.get("base_shift", QWEN2X_BASE_SHIFT))
    base_seq_len = float(opts.get("base_seq_len", QWEN2X_BASE_SEQ_LEN))
    max_seq_len = float(opts.get("max_seq_len", QWEN2X_MAX_SEQ_LEN))
    terminal = float(opts.get("terminal", QWEN2X_SHIFT_TERMINAL))
    stretch = bool(opts.get("stretch", True))

    shift = star_ltx_shift(tokens, max_shift, base_shift, base_seq_len, max_seq_len)
    m = model.clone()
    ms = StarLTXModelSampling(model.model.model_config)
    if stretch:
        f_min = comfy.model_sampling.flux_time_shift(shift, 1.0, 1.0 / max(1, int(steps)))
        ms._ltx_scale = (1.0 - f_min) / (1.0 - terminal)
    ms.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", ms)
    return m, shift


class StarNodes_LTXScheduler_Qwen2x_Options:
    """
    ⭐ Star LTX Scheduler (Qwen Image 2.x)

    Options node for ⭐ StarSampler (Unified), ⭐ Star Qwen2 Outpainter and
    ⭐ Star Flux2 Inpainter. Packages the official Qwen-Image-2.1
    dynamic-shifting schedule as an LTX-style sigma warp and hands it to the
    sampler's "options" input. The sampler then builds the sigma curve
    internally instead of using its widget scheduler.

    Fixes the noisy/grid output at higher resolutions (e.g. 2048x2048): the
    default model sampling uses the fixed shift tuned for 1024x1024
    (mu=0.69), while the official schedule scales the shift with the image
    token count:

        mu = tokens * (max_shift - base_shift) / (max_seq_len - base_seq_len)
             + base_shift - (max_shift - base_shift) / (max_seq_len - base_seq_len) * base_seq_len
        sigma = e^mu / (e^mu + (1/t - 1))

    with the tail stretched so the smallest sigma sits at `terminal`
    (Qwen's `shift_terminal`). Token count is read from the connected
    latent, or from the sampler's own latent when left unconnected, so
    the schedule always matches the actual resolution.
    """
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"

    # Official Qwen-Image-2.1 scheduler config (FlowMatchEulerDiscreteScheduler,
    # use_dynamic_shifting + time_shift_type="exponential").
    MAX_SHIFT = 0.9
    BASE_SHIFT = 0.5
    BASE_SEQ_LEN = 256
    MAX_SEQ_LEN = 8192
    SHIFT_TERMINAL = 0.02

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 40,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of sampling steps (official Qwen-Image-2.1 default is 40). Overrides the steps widget on the connected sampler node.",
                    },
                ),
            },
            "optional": {
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "The latent being sampled - its spatial size defines the image token count used for the dynamic shift. If left unconnected the sampler uses its own latent.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STARNODES_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Sampler"
    DESCRIPTION = "Resolution-aware dynamic-shift sigma schedule (LTX-style) for Qwen Image 2.x. Connect to the options input of ⭐ StarSampler (Unified), ⭐ Star Qwen2 Outpainter or ⭐ Star Flux2 Inpainter."

    def create(self, steps, latent=None):
        # Qwen Image 2.x uses one token per latent pixel (64x64 = 4096 at 1024px).
        # When no latent is connected the sampler computes the token count
        # from its own latent.
        tokens = int(math.prod(latent["samples"].shape[2:])) if latent is not None else None
        payload = {
            "starnodes_type": "LTX_SCHEDULER_QWEN2X",
            "steps": max(1, int(steps)),
            "max_shift": self.MAX_SHIFT,
            "base_shift": self.BASE_SHIFT,
            "stretch": True,
            "terminal": self.SHIFT_TERMINAL,
            "base_seq_len": self.BASE_SEQ_LEN,
            "max_seq_len": self.MAX_SEQ_LEN,
            "tokens": tokens,
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "StarNodes_LTXScheduler_Qwen2x_Options": StarNodes_LTXScheduler_Qwen2x_Options,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarNodes_LTXScheduler_Qwen2x_Options": "⭐ Star LTX Scheduler (Qwen Image 2.x)",
}
