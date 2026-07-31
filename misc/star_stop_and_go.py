import torch
import numpy as np
from PIL import Image
import os
import json
import time
import folder_paths
import threading

# AlwaysEqualProxy allows any type to connect
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True
    
    def __ne__(self, _):
        return False

# ANSI color codes for console output
class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Global state to track user decisions
_user_decisions = {}
_decision_lock = threading.Lock()

def print_starnodes(message, color=Colors.PURPLE):
    """Print a StarNodes message with color and star icon"""
    print(f"{color}{Colors.BOLD}⭐ StarNodes:{Colors.ENDC} {color}{message}{Colors.ENDC}")

class StarStopAndGo:
    """
    Interactive workflow control node with image preview.
    Allows users to stop workflow execution or continue with Stop/Go buttons.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(str(int(time.time())))
        self.compress_level = 4
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["User Select", "Pause", "Bypass"], {
                    "default": "User Select",
                    "tooltip": "User Select: Wait for GO/STOP button click | Pause: Auto-continue after delay | Bypass: Pass through immediately"
                }),
            },
            "optional": {
                "any_input": (AlwaysEqualProxy("*"), {
                    "tooltip": "Connect any data type (IMAGE, LATENT, MODEL, etc.). Image inputs will show preview if enabled."
                }),
                "pause_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 300.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "How long to pause in Pause mode (0.1 to 300 seconds). Only used when mode is set to Pause."
                }),
                "timeout_seconds": ("FLOAT", {
                    "default": 300.0,
                    "min": 10.0,
                    "max": 3600.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Auto-continue timeout for User Select mode (10 to 3600 seconds). Workflow continues automatically if no decision is made within this time."
                }),
                "preview_image": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Show Preview",
                    "label_off": "No Preview",
                    "tooltip": "Show image preview if input is an IMAGE. Disable to skip preview generation for faster processing."
                }),
                "empty_vram_ram": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Clear Memory",
                    "label_off": "Keep Memory",
                    "tooltip": "Clear VRAM before executing. Unloads models, clears cache, and shows memory usage before/after. Useful for freeing memory between operations."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    
    def clear_memory(self):
        """
        Clear VRAM and RAM by unloading models and clearing caches.
        Shows memory usage before and after cleaning.
        """
        try:
            import gc
            import comfy.model_management as mm
            import psutil
            
            # Get memory usage BEFORE cleaning
            process = psutil.Process()
            ram_before = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            
            vram_allocated_before = 0
            vram_reserved_before = 0
            vram_allocated_after = 0
            vram_reserved_after = 0
            vram_total = 0
            
            if torch.cuda.is_available():
                vram_allocated_before = torch.cuda.memory_allocated() / (1024 ** 3)  # Actually allocated
                vram_reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)  # Reserved by PyTorch
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            print_starnodes("Clearing VRAM...", Colors.CYAN)
            
            if torch.cuda.is_available():
                print_starnodes(f"VRAM allocated: {vram_allocated_before:.2f} GB, reserved: {vram_reserved_before:.2f} GB / {vram_total:.2f} GB ({(vram_reserved_before/vram_total*100):.1f}%)", Colors.YELLOW)
            
            # Unload all models from VRAM
            mm.unload_all_models()
            mm.soft_empty_cache()
            
            # Clear Python garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Get memory usage AFTER cleaning
            ram_after = process.memory_info().rss / (1024 ** 3)
            if torch.cuda.is_available():
                vram_allocated_after = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_reserved_after = torch.cuda.memory_reserved() / (1024 ** 3)
            
            # Calculate freed memory
            ram_freed = ram_before - ram_after
            vram_allocated_freed = vram_allocated_before - vram_allocated_after
            vram_reserved_freed = vram_reserved_before - vram_reserved_after
            
            print_starnodes("Memory cleared successfully", Colors.GREEN)
            
            if torch.cuda.is_available():
                print_starnodes(f"VRAM allocated: {vram_allocated_after:.2f} GB, reserved: {vram_reserved_after:.2f} GB / {vram_total:.2f} GB ({(vram_reserved_after/vram_total*100):.1f}%)", Colors.GREEN)
                print_starnodes(f"VRAM freed - allocated: {vram_allocated_freed:.2f} GB, reserved: {vram_reserved_freed:.2f} GB", Colors.GREEN)
            
        except Exception as e:
            print_starnodes(f"Error clearing memory: {e}", Colors.YELLOW)
    
    def process(self, mode, any_input=None, pause_seconds=5.0, timeout_seconds=300.0, 
                preview_image=True, empty_vram_ram=False, prompt=None, extra_pnginfo=None, unique_id=None):
        """
        Process any input based on the selected mode.
        
        Args:
            mode: Operation mode (User Select, Pause, Bypass)
            any_input: Any input type (will preview if it's an image)
            pause_seconds: Seconds to pause in Pause mode
            timeout_seconds: Seconds to wait before auto-continue in User Select mode
            preview_image: Whether to show image preview
            empty_vram_ram: Whether to clear VRAM/RAM
            prompt: ComfyUI prompt data
            extra_pnginfo: Extra PNG info
            unique_id: Unique node ID
            
        Returns:
            Dictionary with input data and optional UI preview
        """
        # Check if input is an image tensor for preview
        results = []
        is_image = isinstance(any_input, torch.Tensor) and any_input.ndim == 4
        
        # Detect input type
        if any_input is None:
            input_type = "None"
            print_starnodes(f"No input connected - Node will act as checkpoint only", Colors.YELLOW)
        elif is_image:
            input_type = "IMAGE"
            if preview_image:
                print_starnodes(f"Input type: IMAGE - Preview will be shown", Colors.YELLOW)
            else:
                print_starnodes(f"Input type: IMAGE - Preview disabled", Colors.YELLOW)
        else:
            input_type = type(any_input).__name__
            print_starnodes(f"Input type: {input_type} - No preview available", Colors.YELLOW)
        
        # Create preview if input is an image and preview is enabled
        if is_image and preview_image:
            for batch_number, img_tensor in enumerate(any_input):
                # Convert from tensor to numpy
                i = 255. * img_tensor.cpu().numpy()
                img_array = np.clip(i, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array)
                
                # Save temporary file for preview
                filename = f"star_stop_go_{unique_id}_{batch_number:05d}.png"
                filepath = os.path.join(self.output_dir, filename)
                pil_image.save(filepath, compress_level=self.compress_level)
                
                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                })
            
            # Send preview to UI immediately before waiting
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("executed", {
                    "node": unique_id,
                    "output": {
                        "images": results
                    },
                    "prompt_id": prompt.get("prompt_id") if prompt else None
                })
                print_starnodes("Preview sent to UI", Colors.GREEN)
            except Exception as e:
                print_starnodes(f"Could not send preview: {e}", Colors.YELLOW)
        
        # Clear memory if requested (before waiting/pausing)
        if empty_vram_ram:
            self.clear_memory()
        
        # Handle different modes
        if mode == "Bypass":
            print_starnodes(f"Mode: Bypass - Passing through immediately", Colors.YELLOW)
            return {
                "ui": {
                    "images": results,
                },
                "result": (any_input,)
            }
        elif mode == "Pause":
            print_starnodes(f"Mode: Pause - Workflow will pause for {pause_seconds} seconds", Colors.YELLOW)
            time.sleep(pause_seconds)
            print_starnodes(f"Pause complete - Continuing workflow", Colors.YELLOW)
            return {
                "ui": {
                    "images": results,
                },
                "result": (any_input,)
            }
        else:  # User Select mode
            print_starnodes(f"Mode: User Select - Workflow paused, waiting for your decision", Colors.YELLOW)
            print_starnodes("Click GO to continue or STOP to halt the workflow", Colors.CYAN)
            
            # Display timeout in a user-friendly format
            timeout_minutes = int(timeout_seconds // 60)
            timeout_remaining_seconds = int(timeout_seconds % 60)
            if timeout_minutes > 0 and timeout_remaining_seconds > 0:
                timeout_display = f"{timeout_minutes} min {timeout_remaining_seconds} sec"
            elif timeout_minutes > 0:
                timeout_display = f"{timeout_minutes} min"
            else:
                timeout_display = f"{int(timeout_seconds)} sec"
            
            print_starnodes(f"Auto-continue timeout: {timeout_display} ({int(timeout_seconds)} seconds)", Colors.CYAN)
            
            # Store that we're waiting for this node
            with _decision_lock:
                _user_decisions[unique_id] = "waiting"
            
            # Wait for user decision with timeout
            max_wait_time = timeout_seconds
            poll_interval = 0.5  # Check every 0.5 seconds
            elapsed = 0
            
            while elapsed < max_wait_time:
                time.sleep(poll_interval)
                elapsed += poll_interval
                
                with _decision_lock:
                    decision = _user_decisions.get(unique_id, "waiting")
                
                if decision == "go":
                    print_starnodes("User clicked GO - Continuing workflow", Colors.GREEN)
                    with _decision_lock:
                        if unique_id in _user_decisions:
                            del _user_decisions[unique_id]
                    return {
                        "ui": {
                            "images": results,
                        },
                        "result": (any_input,)
                    }
                elif decision == "stop":
                    print_starnodes("Workflow stopped by user", Colors.PURPLE)
                    with _decision_lock:
                        if unique_id in _user_decisions:
                            del _user_decisions[unique_id]
                    
                    # The frontend already called api.interrupt() when the button was clicked
                    # Just wait a moment for the interrupt to take effect, then return
                    # This prevents the exception traceback from showing
                    time.sleep(0.5)
                    
                    # Return the input - the interrupt will stop execution before next node
                    return {
                        "ui": {
                            "images": results,
                        },
                        "result": (any_input,)
                    }
            
            # Timeout - default to continue
            # Display timeout reached message with user-friendly format
            if timeout_minutes > 0 and timeout_remaining_seconds > 0:
                timeout_msg = f"{timeout_minutes} min {timeout_remaining_seconds} sec"
            elif timeout_minutes > 0:
                timeout_msg = f"{timeout_minutes} min"
            else:
                timeout_msg = f"{int(timeout_seconds)} sec"
            
            print_starnodes(f"Timeout reached ({timeout_msg}) - Continuing workflow automatically", Colors.YELLOW)
            with _decision_lock:
                if unique_id in _user_decisions:
                    del _user_decisions[unique_id]
            return {
                "ui": {
                    "images": results,
                },
                "result": (any_input,)
            }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to allow for interactive control
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "StarStopAndGo": StarStopAndGo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarStopAndGo": "⭐ Star Stop And Go"
}

# API endpoint for receiving user decisions
try:
    from server import PromptServer
    from aiohttp import web
    
    @PromptServer.instance.routes.post("/starnodes/stop_and_go/decision")
    async def set_user_decision(request):
        try:
            data = await request.json()
            node_id = data.get("node_id")
            decision = data.get("decision", "waiting")
            
            if node_id:
                with _decision_lock:
                    _user_decisions[node_id] = decision
                # Don't print here - let the main process method handle the colored output
                return web.json_response({"status": "ok", "node_id": node_id, "decision": decision})
            else:
                return web.json_response({"status": "error", "message": "node_id required"}, status=400)
        except Exception as e:
            print_starnodes(f"API Error: {e}", Colors.RED)
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    print_starnodes(f"Could not register API endpoint: {e}", Colors.RED)
