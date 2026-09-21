import torch

try:
    import folder_paths
    from folder_paths import get_output_directory
except ImportError:
    import os

    folder_paths = None

    def get_output_directory():
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        return out


def install_lazy_quant_save_fix():
    """Make ComfyUI's lazy checkpoint-save wrappers accept integer-dtype tensors.

    NVFP4 (and similar 4-bit) layouts store the packed quantized weight as uint8.
    ComfyUI wraps saved weights in `torch.nn.Parameter` subclasses whose __new__
    defaults to requires_grad=True, which PyTorch rejects for non-float dtypes:
    "Only Tensors of floating point and complex dtype can require gradients".

    This fix lives here (instead of only in the ComfyUI fork) so AIO saving keeps
    working even after a ComfyUI update overwrites core files. It is idempotent,
    only active for non-float/non-complex tensors, and silently skipped if a
    future ComfyUI renames or removes the wrapper classes.
    """
    try:
        from comfy import model_patcher
    except ImportError:
        return

    base_new = torch.nn.Parameter.__new__

    def wrap(cls):
        orig = cls.__new__
        if getattr(orig, "_star_quant_fix", False):
            return

        def __new__(cls_, *args, **kwargs):
            tensor = args[-1]
            if isinstance(tensor, torch.Tensor) and not (
                torch.is_floating_point(tensor) or torch.is_complex(tensor)
            ):
                return base_new(cls_, tensor, requires_grad=False)
            return orig(cls_, *args, **kwargs)

        __new__._star_quant_fix = True
        cls.__new__ = staticmethod(__new__)

    for name in ("LazyCastingParam", "LazyCastingParamPiece"):
        cls = model_patcher.__dict__.get(name)
        if cls is not None:
            wrap(cls)


install_lazy_quant_save_fix()


class StarCheckpointSaver:
    def __init__(self):
        self.output_dir = get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "checkpoints/StarCheckpointSave"}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = "Saves model+clip+vae as one AIO checkpoint. Works with mixed quantizations (e.g. NVFP4 model + NVFP4 text encoder + BF16 VAE)."

    def save(self, model, clip, vae, filename_prefix, clip_vision=None, prompt=None, extra_pnginfo=None):
        install_lazy_quant_save_fix()
        from comfy_extras.nodes_model_merging import save_checkpoint
        save_checkpoint(model, clip=clip, vae=vae, clip_vision=clip_vision,
                        filename_prefix=filename_prefix, output_dir=self.output_dir,
                        prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}


NODE_CLASS_MAPPINGS = {
    "StarCheckpointSaver": StarCheckpointSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarCheckpointSaver": "⭐ Star Checkpoint Saver (AIO)",
}
