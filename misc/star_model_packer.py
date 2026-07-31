import os
import glob
import torch
from safetensors.torch import load_file, save_file

try:
    from folder_paths import get_output_directory
except ImportError:
    def get_output_directory():
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        return out


class StarModelPacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Full path to the folder containing split safetensors files
                "input_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "D:/AI/ComfyUI202511/HF/qwen3-4b-heretic-zimage/qwen-4b-zimage-heretic",
                    },
                ),
                # Floating point precision selection
                "precision": (
                    ["FP8", "FP16", "FP32"],
                    {
                        "default": "FP8",
                    },
                ),
                # Optional custom save name (if empty, uses folder name + precision)
                "save_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave empty to auto-generate from folder name",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "pack_model"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def pack_model(self, input_folder: str, precision: str, save_name: str):
        input_folder = (input_folder or "").strip()
        if not input_folder:
            return ("No input folder provided.",)

        # Normalize and validate source path
        src_path = os.path.abspath(os.path.expanduser(input_folder))
        if not os.path.isdir(src_path):
            return (f"Input folder not found: {src_path}",)

        # Find all safetensors files in the folder
        pattern = os.path.join(src_path, "*.safetensors")
        safetensor_files = sorted(glob.glob(pattern))
        
        if not safetensor_files:
            return (f"No .safetensors files found in: {src_path}",)

        print(f"\n{'='*60}")
        print(f"Star Model Packer - Starting")
        print(f"{'='*60}")
        print(f"Found {len(safetensor_files)} safetensors file(s)")
        for i, f in enumerate(safetensor_files, 1):
            print(f"  {i}. {os.path.basename(f)}")

        # Determine output filename
        save_name = (save_name or "").strip()
        if not save_name:
            # Use folder name + precision suffix
            folder_name = os.path.basename(src_path)
            save_name = f"{folder_name}_{precision}"
        
        # Ensure .safetensors extension
        if not save_name.lower().endswith(".safetensors"):
            save_name = save_name + ".safetensors"

        # Output goes to standard output directory under /models/
        out_root = get_output_directory()
        models_out = os.path.join(out_root, "models")
        os.makedirs(models_out, exist_ok=True)
        dst_path = os.path.join(models_out, save_name)

        print(f"\nTarget precision: {precision}")
        print(f"Output file: {dst_path}")
        print(f"\n{'='*60}")
        print("Loading and combining tensors...")
        print(f"{'='*60}")

        # Load all tensors from all files
        combined_tensors = {}
        total_tensors = 0
        
        try:
            for idx, file_path in enumerate(safetensor_files, 1):
                print(f"\nLoading file {idx}/{len(safetensor_files)}: {os.path.basename(file_path)}")
                tensors = load_file(file_path)
                print(f"  Found {len(tensors)} tensor(s)")
                
                # Check for duplicate keys
                for key in tensors.keys():
                    if key in combined_tensors:
                        print(f"  Warning: Duplicate key '{key}' found, overwriting...")
                    combined_tensors[key] = tensors[key]
                    total_tensors += 1
                
                print(f"  Progress: {total_tensors} total tensors loaded")
                
        except Exception as e:
            return (f"Failed to load safetensors files: {e}",)

        print(f"\n{'='*60}")
        print(f"Converting to {precision}...")
        print(f"{'='*60}")

        # Convert tensors to target precision
        converted_tensors = {}
        conversion_count = 0
        skipped_count = 0
        
        # Map precision string to torch dtype
        precision_map = {
            "FP8": torch.float8_e4m3fn,
            "FP16": torch.float16,
            "FP32": torch.float32,
        }
        target_dtype = precision_map[precision]

        try:
            for name, tensor in combined_tensors.items():
                conversion_count += 1
                if conversion_count % 100 == 0:
                    print(f"  Converting tensor {conversion_count}/{total_tensors}...")
                
                # Convert floating point tensors to target precision
                if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    try:
                        converted_tensors[name] = tensor.to(target_dtype)
                    except Exception as e:
                        print(f"  Warning: Failed to convert tensor '{name}' to {precision}: {e}")
                        print(f"  Keeping original dtype: {tensor.dtype}")
                        converted_tensors[name] = tensor
                        skipped_count += 1
                else:
                    # Keep non-floating point tensors as-is
                    converted_tensors[name] = tensor
                    skipped_count += 1
                    
        except Exception as e:
            return (f"Failed to convert tensors to {precision}: {e}",)

        print(f"\nConversion complete:")
        print(f"  Converted: {conversion_count - skipped_count} tensors")
        print(f"  Skipped: {skipped_count} tensors (non-floating or failed)")

        print(f"\n{'='*60}")
        print("Saving packed model...")
        print(f"{'='*60}")

        try:
            save_file(converted_tensors, dst_path)
            print(f"Successfully saved to: {dst_path}")
        except Exception as e:
            return (f"Failed to save packed model: {e}",)

        # Calculate file sizes
        try:
            input_size = sum(os.path.getsize(f) for f in safetensor_files)
        except Exception:
            input_size = -1
        
        try:
            output_size = os.path.getsize(dst_path)
        except Exception:
            output_size = -1

        def _fmt_size(bytes_val: int) -> str:
            if bytes_val < 0:
                return "unknown"
            mb = bytes_val / (1024 * 1024)
            gb = bytes_val / (1024 * 1024 * 1024)
            if gb >= 1.0:
                return f"{gb:.2f} GB ({mb:.2f} MB)"
            return f"{mb:.2f} MB"

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Input files: {len(safetensor_files)}")
        print(f"Total input size: {_fmt_size(input_size)}")
        print(f"Output size: {_fmt_size(output_size)}")
        print(f"Precision: {precision}")
        print(f"Total tensors: {total_tensors}")
        print(f"{'='*60}\n")

        status = (
            f"Model packed successfully!\n"
            f"Output: {os.path.basename(dst_path)}\n"
            f"Saved to: {dst_path}\n"
            f"Input files: {len(safetensor_files)}\n"
            f"Total input size: {_fmt_size(input_size)}\n"
            f"Output size: {_fmt_size(output_size)}\n"
            f"Precision: {precision}\n"
            f"Total tensors: {total_tensors}"
        )

        return (status,)


NODE_CLASS_MAPPINGS = {
    "StarModelPacker": StarModelPacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarModelPacker": "⭐ Star Model Packer",
}
