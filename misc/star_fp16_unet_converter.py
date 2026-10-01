import gc
import json
import os

import torch
from safetensors.torch import save_file

import folder_paths
import comfy.utils


FP8_DTYPES = [
    dt for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
]

FP8_QUANT_FORMATS = {}
if hasattr(torch, "float8_e4m3fn"):
    FP8_QUANT_FORMATS["float8_e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    FP8_QUANT_FORMATS["float8_e5m2"] = torch.float8_e5m2


class StarFP16UnetConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Pick one of the models from the diffusion_models folder (same list as the UNET Loader)
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {}),
                # CPU = low memory footprint (default), GPU = faster conversion on the graphics card
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    @staticmethod
    def _collect_quant_layers(sd, metadata):
        # Returns a dict of layer name -> quantization config and removes the
        # metadata markers from the state dict so they are not saved again.
        layers = {}

        if metadata and "_quantization_metadata" in metadata:
            qm = json.loads(metadata["_quantization_metadata"])
            for layer, conf in qm.get("layers", {}).items():
                layers[layer] = dict(conf)

        for k in list(sd.keys()):
            if k.endswith(".comfy_quant"):
                conf = json.loads(sd[k].numpy().tobytes().decode("utf-8"))
                layers[k[:-len(".comfy_quant")]] = conf
                del sd[k]

        scaled_key = None
        for k in sd.keys():
            if k == "scaled_fp8" or k.endswith(".scaled_fp8"):
                scaled_key = k
                break

        if scaled_key is not None:
            prefix = scaled_key[:-len("scaled_fp8")]
            del sd[scaled_key]
            for k in list(sd.keys()):
                if not k.startswith(prefix):
                    continue
                if k.endswith(".scale_weight"):
                    layer = k[:-len(".scale_weight")]
                    w = sd.get(layer + ".weight")
                    fmt = "float8_e4m3fn"
                    if w is not None and hasattr(torch, "float8_e5m2") and w.dtype == torch.float8_e5m2:
                        fmt = "float8_e5m2"
                    layers[layer] = {"format": fmt}
                    sd[layer + ".weight_scale"] = sd.pop(k)
                elif k.endswith(".scale_input"):
                    del sd[k]

        return layers

    @staticmethod
    def _dequantize_weight(weight, fmt, scale, work_device):
        if fmt in FP8_QUANT_FORMATS:
            storage_dtype = FP8_QUANT_FORMATS[fmt]
            w = weight
            if w.dtype == torch.uint8:
                w = w.view(storage_dtype)
            w = w.to(work_device).to(torch.float32)
            if scale is not None:
                w = w * scale.to(device=work_device, dtype=torch.float32)
            return w.to(torch.float16).cpu()
        if fmt == "int8_tensorwise":
            if scale is None:
                raise ValueError("int8_tensorwise layer is missing its weight_scale")
            w = weight.to(work_device).to(torch.float32) * scale.to(device=work_device, dtype=torch.float32)
            return w.to(torch.float16).cpu()
        raise ValueError(f"quantization format '{fmt}' is not supported by this converter")

    def convert(self, model_name: str, device: str):
        if not model_name:
            return ("No model selected.",)

        try:
            src_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        except Exception as e:
            return (f"Model not found: {e}",)

        src_base = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(os.path.dirname(src_path), f"{src_base}_FP16.safetensors")

        if os.path.isfile(dst_path):
            return (f"Output file already exists, nothing was converted:\n{dst_path}\nRename or delete it first if you want to convert again.",)

        print(f"\n{'='*60}")
        print("Star FP16 Unet Converter - Starting")
        print(f"{'='*60}")
        print(f"Source: {src_path}")
        print(f"Target: {dst_path}")

        if device == "GPU":
            if not torch.cuda.is_available():
                return ("GPU mode selected, but no CUDA device is available. Please use CPU mode.",)
            work_device = torch.device("cuda")
        else:
            work_device = torch.device("cpu")
        print(f"Compute device: {work_device}")

        try:
            sd, metadata = comfy.utils.load_torch_file(src_path, safe_load=True, return_metadata=True)
        except Exception as e:
            return (f"Failed to load model file: {e}",)

        if not isinstance(sd, dict) or len(sd) == 0:
            return ("The selected file does not contain a model state dict, nothing to convert.",)

        try:
            quant_layers = self._collect_quant_layers(sd, metadata)
        except Exception as e:
            return (f"Failed to read quantization metadata: {e}",)

        dequantized_layers = 0
        try:
            for layer, conf in quant_layers.items():
                fmt = conf.get("format", None)
                if conf.get("convrot", False):
                    return (f"Layer '{layer}' uses a rotated quantization ({fmt}, convrot) which this node cannot dequantize to FP16. Conversion aborted.",)
                wkey = layer + ".weight"
                if wkey not in sd:
                    print(f"  Warning: quantized layer '{layer}' has no weight key, skipping")
                    continue
                scale = sd.pop(layer + ".weight_scale", None)
                try:
                    sd[wkey] = self._dequantize_weight(sd[wkey], fmt, scale, work_device)
                except ValueError as e:
                    return (f"Layer '{layer}': {e}. Conversion aborted.",)
                dequantized_layers += 1
                if dequantized_layers % 100 == 0:
                    print(f"  Dequantized {dequantized_layers}/{len(quant_layers)} layers...")
        except Exception as e:
            return (f"Failed to dequantize layer weights: {e}",)

        # Keys that must not be carried over into a plain fp16 checkpoint
        drop_suffixes = (".comfy_quant", ".input_scale", ".scale_input", ".pre_quant_scale", ".weight_scale_2")

        print(f"{'='*60}")
        print("Converting remaining tensors to FP16...")
        print(f"{'='*60}")

        out_sd = {}
        converted_count = 0
        skipped_count = 0
        drop_count = 0
        keys = list(sd.keys())
        total = len(keys)
        try:
            for idx, k in enumerate(keys, 1):
                t = sd.pop(k)
                if idx % 500 == 0:
                    print(f"  Processing tensor {idx}/{total}...")

                if k.endswith(drop_suffixes) or k == "scaled_fp8" or k.endswith(".scaled_fp8"):
                    del t
                    drop_count += 1
                    continue

                if t.dtype in FP8_DTYPES or (t.is_floating_point() and t.dtype != torch.float16):
                    if t.device != work_device:
                        t = t.to(work_device)
                    t = t.to(torch.float16).cpu()
                    converted_count += 1
                elif t.is_floating_point():
                    t = t.cpu()
                    skipped_count += 1
                else:
                    t = t.clone().cpu()
                    skipped_count += 1

                out_sd[k] = t
        except Exception as e:
            return (f"Failed to convert tensor '{k}' to FP16: {e}",)

        del sd
        gc.collect()
        if work_device.type == "cuda":
            torch.cuda.empty_cache()

        out_metadata = None
        if metadata:
            out_metadata = {k: v for k, v in metadata.items() if k != "_quantization_metadata"}
            if len(out_metadata) == 0:
                out_metadata = None

        print(f"{'='*60}")
        print("Saving FP16 model...")
        print(f"{'='*60}")

        try:
            save_file(out_sd, dst_path, metadata=out_metadata)
        except Exception as e:
            try:
                if os.path.isfile(dst_path):
                    os.remove(dst_path)
            except Exception:
                pass
            return (f"Failed to save FP16 checkpoint: {e}",)

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
            gb = bytes_val / (1024 * 1024 * 1024)
            if gb >= 1.0:
                return f"{gb:.2f} GB"
            return f"{bytes_val / (1024 * 1024):.2f} MB"

        print(f"\nDone:")
        print(f"  Dequantized layers: {dequantized_layers}")
        print(f"  Tensors converted to FP16: {converted_count}")
        print(f"  Tensors kept as-is: {skipped_count}")
        print(f"  Quantization metadata entries removed: {drop_count}")
        print(f"  {'='*58}\n")

        status = (
            f"Model converted to FP16!\n"
            f"Saved to: {dst_path}\n"
            f"Old size of file: {_fmt_size(old_size)}\n"
            f"New size of file: {_fmt_size(new_size)}\n"
            f"Dequantized layers: {dequantized_layers}\n"
            f"Tensors converted to FP16: {converted_count}\n"
            f"Tensors kept as-is: {skipped_count}\n"
            f"Compute device used: {work_device}\n"
            f"Restart ComfyUI or press 'r' in the model dropdown to see the new file in the loaders."
        )

        return (status,)


NODE_CLASS_MAPPINGS = {
    "StarFP16UnetConverter": StarFP16UnetConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFP16UnetConverter": "⭐ Star FP16 Unet Converter",
}
