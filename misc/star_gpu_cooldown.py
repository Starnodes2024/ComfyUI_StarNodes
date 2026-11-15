import gc
import time

import torch

import comfy.model_management as mm


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


def print_starnodes(message):
    print(f"⭐ StarNodes: {message}")


def _vram_stats():
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return allocated, reserved, total


def _format_vram(stats):
    if stats is None:
        return "VRAM: CUDA not available"
    allocated, reserved, total = stats
    return f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB / {total:.2f} GB"


def _aggressive_cleanup(clean_ram, label):
    """Unload all models, flush the CUDA allocator and (optionally) free RAM."""
    print_starnodes(f"{label} - before: {_format_vram(_vram_stats())}")

    # Unload every model from VRAM and clear caches
    mm.unload_all_models()
    mm.soft_empty_cache(True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if clean_ram:
        # Full garbage collection frees unreferenced Python objects in RAM
        gc.collect()
        gc.collect()
        gc.collect()
    else:
        gc.collect()

    print_starnodes(f"{label} - after: {_format_vram(_vram_stats())}")


class StarGPUCooldown:
    """
    Empties the GPU VRAM completely (unloads all models, clears the whole
    cache), waits a configurable cooldown time, performs a second aggressive
    cleanup and then continues the workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wait_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 600.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Cooldown time in seconds between the first full VRAM cleanup and the second aggressive cleanup."
                }),
                "clean_ram": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Clean RAM too",
                    "label_off": "VRAM only",
                    "tooltip": "Additionally free RAM: runs a full Python garbage collection after each cleanup step."
                }),
            },
            "optional": {
                "any_input": (AlwaysEqualProxy("*"), {
                    "tooltip": "Optional: connect any data type (IMAGE, LATENT, MODEL, ...) to pass it through unchanged after the cooldown."
                }),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = ("Cooldown for the GPU: unloads all models and fully clears the VRAM cache, "
                   "waits the configured time, runs a second aggressive cleanup and then "
                   "continues the workflow. Optionally also frees RAM.")

    def process(self, wait_seconds, clean_ram, any_input=None):
        print_starnodes("GPU Cooldown started")

        # Step 1: full VRAM cleanup
        _aggressive_cleanup(clean_ram, "Cleanup 1/2")

        # Step 2: cooldown waiting time
        if wait_seconds > 0:
            print_starnodes(f"Cooling down for {wait_seconds:.0f} seconds...")
            time.sleep(wait_seconds)

        # Step 3: second aggressive cleanup to catch anything freed late
        _aggressive_cleanup(clean_ram, "Cleanup 2/2")

        print_starnodes("GPU Cooldown finished - continuing workflow")
        return (any_input,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so the cooldown runs on every queue
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "StarGPUCooldown": StarGPUCooldown,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarGPUCooldown": "⭐ Star GPU Cooldown",
}
