import os

import torch
from safetensors.torch import load_file, save_file

try:
    from folder_paths import get_output_directory
except ImportError:
    def get_output_directory():
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        return out


class StarFP8Converter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Full path to the source .safetensors file (e.g. F:/ComfyUIModels/models/clip/qwen_3_4b.safetensors)
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "F:/ComfyUIModels/models/clip/qwen_3_4b.safetensors",
                    },
                ),
                # Desired base name for the converted file (extension .safetensors will be added if missing)
                "save_name": (
                    "STRING",
                    {
                        "default": "_fp8_scaled_e4m3fn.safetensors",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def convert(self, model_path: str, save_name: str):
        model_path = (model_path or "").strip()
        if not model_path:
            return ("No model path provided.",)

        # Normalize and validate source path
        src_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.isfile(src_path):
            return (f"Source file not found: {src_path}",)

        save_name = (save_name or "").strip()
        if not save_name:
            return ("Save name is empty.",)

        # Default behavior: if the user keeps the predefined suffix, build
        # the filename from the original basename plus that suffix.
        default_suffix = "_fp8_scaled_e4m3fn.safetensors"
        src_base = os.path.splitext(os.path.basename(src_path))[0]
        if save_name == default_suffix:
            final_filename = f"{src_base}{default_suffix}"
        else:
            # Treat save_name as an explicit filename, ensure .safetensors extension
            if not save_name.lower().endswith(".safetensors"):
                save_name = save_name + ".safetensors"
            final_filename = save_name

        # Output goes to standard output directory under /models/
        out_root = get_output_directory()
        models_out = os.path.join(out_root, "models")
        os.makedirs(models_out, exist_ok=True)
        dst_path = os.path.join(models_out, final_filename)

        try:
            tensors = load_file(src_path)
        except Exception as e:
            return (f"Failed to load source safetensors: {e}",)

        new_tensors = {}
        for name, t in tensors.items():
            # Convert all floating tensors to fp8_e4m3fn, keep non-floats as-is
            try:
                if hasattr(t, "dtype") and getattr(t.dtype, "is_floating_point", False):
                    t_fp8 = t.to(torch.float8_e4m3fn)
                    new_tensors[name] = t_fp8
                else:
                    new_tensors[name] = t
            except Exception as e:
                return (f"Failed to convert tensor '{name}' to fp8_e4m3fn: {e}",)

        try:
            save_file(new_tensors, dst_path)
        except Exception as e:
            return (f"Failed to save FP8 checkpoint: {e}",)

        # Build detailed status including sizes
        try:
            old_size = os.path.getsize(src_path)
        except Exception:
            old_size = -1
        try:
            new_size = os.path.getsize(dst_path)
        except Exception:
            new_size = -1

        def _fmt_size(bytes_val: int) -> str:
            if bytes_val < 0:
                return "unknown"
            mb = bytes_val / (1024 * 1024)
            return f"{mb:.2f} MB ({bytes_val} bytes)"

        status = (
            f"Model converted to: {os.path.basename(dst_path)}\n"
            f"Saved to: {dst_path}\n"
            f"Old size of file: {_fmt_size(old_size)}\n"
            f"New size of file: {_fmt_size(new_size)}"
        )

        return (status,)


NODE_CLASS_MAPPINGS = {
    "StarFP8Converter": StarFP8Converter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFP8Converter": "⭐ Star FP8 Converter",
}
