import torch
import os
import sys
import folder_paths
import comfy.utils
import comfy.sd

class Star3LoRAs:
    CATEGORY = '⭐StarNodes'
    
    def __init__(self):
        self.loaded_loras = [None, None, None]

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        # Add "None" as the first option for each LoRA
        lora_list = ["None"] + lora_list
        
        return {
            "required": {
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
                
                # LoRA 1 settings
                "lora1_name": (lora_list, {"tooltip": "The name of the first LoRA to apply. Select 'None' to skip."}),
                "strength1_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model with the first LoRA."}),
                "strength1_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model with the first LoRA."}),
                
                # LoRA 2 settings
                "lora2_name": (lora_list, {"default": "None", "tooltip": "The name of the second LoRA to apply. Select 'None' to skip."}),
                "strength2_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model with the second LoRA."}),
                "strength2_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model with the second LoRA."}),
                
                # LoRA 3 settings
                "lora3_name": (lora_list, {"default": "None", "tooltip": "The name of the third LoRA to apply. Select 'None' to skip."}),
                "strength3_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model with the third LoRA."}),
                "strength3_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model with the third LoRA."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRAs will be applied to. Optional."}),
                "LoRA_Stack": ("LORA_STACK", {"tooltip": "Optional input for chaining multiple LoRA nodes together."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK")
    RETURN_NAMES = ("model", "clip", "lora_stack")
    OUTPUT_IS_LIST = (False, False, False)
    OUTPUT_TOOLTIPS = (
        "The modified diffusion model with all LoRAs applied.", 
        "The modified CLIP model with all LoRAs applied (if provided).",
        "A stack of all applied LoRAs for use with FluxStart or chaining to another LoRA node."
    )
    FUNCTION = "apply_loras"
    DESCRIPTION = "Apply up to three LoRAs sequentially to a model and optionally to CLIP. Each LoRA can be individually configured or disabled. Can be connected to FluxStart for internal conditioning."

    def load_lora(self, lora_name, lora_index):
        """Helper function to load a LoRA file"""
        if lora_name == "None":
            return None
            
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        # Check if we already have this LoRA loaded
        if self.loaded_loras[lora_index] is not None:
            if self.loaded_loras[lora_index][0] == lora_path:
                return self.loaded_loras[lora_index][1]
            else:
                self.loaded_loras[lora_index] = None
        
        # Load the LoRA
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_index] = (lora_path, lora)
        return lora

    def apply_loras(self, model=None, lora1_name="None", strength1_model=1.0, strength1_clip=1.0, 
                   lora2_name="None", strength2_model=1.0, strength2_clip=1.0,
                   lora3_name="None", strength3_model=1.0, strength3_clip=1.0,
                   clip=None, LoRA_Stack=None):
        
        model_out = model
        clip_out = clip
        
        # Initialize lora_stack from input or create a new one
        lora_stack = LoRA_Stack if LoRA_Stack is not None else []
        
        # Apply LoRA 1 if selected
        if lora1_name != "None" and strength1_model != 0:
            lora1 = self.load_lora(lora1_name, 0)
            if lora1 is not None:
                if model_out is not None:
                    model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora1, strength1_model, strength1_clip if clip_out is not None else 0)
                # Add to stack
                lora_stack.append({
                    "name": lora1_name,
                    "model_strength": strength1_model,
                    "clip_strength": strength1_clip,
                    "lora": lora1
                })
        
        # Apply LoRA 2 if selected
        if lora2_name != "None" and strength2_model != 0:
            lora2 = self.load_lora(lora2_name, 1)
            if lora2 is not None:
                if model_out is not None:
                    model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora2, strength2_model, strength2_clip if clip_out is not None else 0)
                # Add to stack
                lora_stack.append({
                    "name": lora2_name,
                    "model_strength": strength2_model,
                    "clip_strength": strength2_clip,
                    "lora": lora2
                })
        
        # Apply LoRA 3 if selected
        if lora3_name != "None" and strength3_model != 0:
            lora3 = self.load_lora(lora3_name, 2)
            if lora3 is not None:
                if model_out is not None:
                    model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora3, strength3_model, strength3_clip if clip_out is not None else 0)
                # Add to stack
                lora_stack.append({
                    "name": lora3_name,
                    "model_strength": strength3_model,
                    "clip_strength": strength3_clip,
                    "lora": lora3
                })
        
        return (model_out, clip_out, lora_stack)

# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "Star3LoRAs": Star3LoRAs
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "Star3LoRAs": "⭐Star 3 LoRAs"
}
