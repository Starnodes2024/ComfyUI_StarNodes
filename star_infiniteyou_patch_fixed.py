import torch
import os
import comfy.utils
import folder_paths
import numpy as np

# Ensure output directory exists for patch files
PATCH_DIR = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
os.makedirs(PATCH_DIR, exist_ok=True)

# Star InfiniteYou Patch Loader node implementation

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarInfiniteYouPatch": "⭐StarNodes/InfiniteYou/Patch Loader",
    "StarInfiniteYouPatchCombine": "⭐StarNodes/InfiniteYou/Patch Combine"
}

class StarInfiniteYouPatch:
    @classmethod
    def INPUT_TYPES(s):
        # Get list of patch files
        patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(patch_dir, exist_ok=True)
        
        # Get both .pt and .iyou patch files (for compatibility)
        patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
        # Sort patches alphabetically for better organization
        patches.sort()
        
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "patch_file": (["none"] + patches,),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_patch"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def apply_patch(self, control_net, model, positive, negative, vae, latent_image, patch_file):
        if patch_file == "none":
            # If no patch file is selected, just return the inputs unchanged
            return (model, positive, negative, latent_image)
        
        # Load the patch data
        patch_path = os.path.join(folder_paths.output_directory, "infiniteyoupatch", patch_file)
        patch_data = torch.load(patch_path)
        
        # Get the embeddings and parameters from the patch data
        device = comfy.model_management.get_torch_device()
        image_prompt_embeds = patch_data["image_prompt_embeds"].to(device) if "image_prompt_embeds" in patch_data else None
        uncond_image_prompt_embeds = patch_data["uncond_image_prompt_embeds"].to(device) if "uncond_image_prompt_embeds" in patch_data else None
        face_kps = patch_data["face_kps"] if "face_kps" in patch_data else None
        
        # Get control parameters
        cn_strength = patch_data.get("cn_strength", 1.0)
        start_at = patch_data.get("start_at", 0.0)
        end_at = patch_data.get("end_at", 1.0)
        
        # Make sure we have all the necessary data
        if image_prompt_embeds is None or uncond_image_prompt_embeds is None or face_kps is None:
            print("Warning: Patch file is missing essential data. Cannot apply face.")
            return (model, positive, negative, latent_image)
        
        print(f"Applying face from patch file: {patch_file} to user conditioning")
        
        # Set up the controlnet with the face keypoints
        cnets = {}
        cond_uncond = []
        
        # Process positive and negative conditioning separately
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                # Get the original weight and conditioning dictionary
                weight, cond_dict = t
                
                # Create a copy of the conditioning dictionary to avoid modifying the original
                d = cond_dict.copy()
                
                # Set up the control net
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                # Apply only the face-specific parts to the conditioning
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                
                # Add the face embeddings
                if is_cond:
                    d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)
                else:
                    d['cross_attn_controlnet'] = uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)
                
                # Create the new conditioning entry with the original weight
                n = [weight, d]
                c.append(n)
            
            # Add the processed conditioning to the appropriate list
            cond_uncond.append(c)
            is_cond = False
        
        return(model, cond_uncond[0], cond_uncond[1], latent_image)


class StarInfiniteYouPatchCombine:
    @classmethod
    def INPUT_TYPES(s):
        # Get list of patch files
        patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(patch_dir, exist_ok=True)
        
        # Get both .pt and .iyou patch files (for compatibility)
        patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
        # Sort patches alphabetically for better organization
        patches.sort()
        
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "patch_file_1": (["none"] + patches,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_2": (["none"] + patches,),
                "weight_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "patch_file_3": (["none"] + patches,),
                "weight_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_4": (["none"] + patches,),
                "weight_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_5": (["none"] + patches,),
                "weight_5": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_combined_patches"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def load_patch(self, patch_file):
        if patch_file == "none":
            return None
        
        # Load the patch data
        patch_path = os.path.join(folder_paths.output_directory, "infiniteyoupatch", patch_file)
        patch_data = torch.load(patch_path)
        
        # Get the embeddings and parameters from the patch data
        device = comfy.model_management.get_torch_device()
        image_prompt_embeds = patch_data["image_prompt_embeds"].to(device) if "image_prompt_embeds" in patch_data else None
        uncond_image_prompt_embeds = patch_data["uncond_image_prompt_embeds"].to(device) if "uncond_image_prompt_embeds" in patch_data else None
        face_kps = patch_data["face_kps"] if "face_kps" in patch_data else None
        
        # Return None if missing essential data
        if image_prompt_embeds is None or uncond_image_prompt_embeds is None or face_kps is None:
            print(f"Warning: Patch file {patch_file} is missing essential data.")
            return None
            
        return {
            "file_name": patch_file,
            "image_prompt_embeds": image_prompt_embeds,
            "uncond_image_prompt_embeds": uncond_image_prompt_embeds,
            "face_kps": face_kps
        }
    
    def apply_combined_patches(self, control_net, model, positive, negative, vae, latent_image, 
                             patch_file_1, weight_1, patch_file_2, weight_2,
                             patch_file_3=None, weight_3=0.0, patch_file_4=None, weight_4=0.0, 
                             patch_file_5=None, weight_5=0.0, cn_strength=1.0, start_at=0.0, end_at=1.0):
        
        # Check if at least one patch file is selected
        if patch_file_1 == "none" and patch_file_2 == "none" and \
           (patch_file_3 is None or patch_file_3 == "none") and \
           (patch_file_4 is None or patch_file_4 == "none") and \
           (patch_file_5 is None or patch_file_5 == "none"):
            # If no patch file is selected, just return the inputs unchanged
            return (model, positive, negative, latent_image)
        
        # Load all patch files
        patches = []
        weights = []
        
        # Load each patch with its weight, if valid
        patch_files = [(patch_file_1, weight_1), (patch_file_2, weight_2)]
        if patch_file_3 is not None:
            patch_files.append((patch_file_3, weight_3))
        if patch_file_4 is not None:
            patch_files.append((patch_file_4, weight_4))
        if patch_file_5 is not None:
            patch_files.append((patch_file_5, weight_5))
            
        loaded_files = []
        for patch_file, weight in patch_files:
            if patch_file != "none" and weight > 0.0:
                patch = self.load_patch(patch_file)
                if patch is not None:
                    patches.append(patch)
                    weights.append(weight)
                    loaded_files.append(patch_file)
        
        if not patches:
            print("No valid patches to combine.")
            return (model, positive, negative, latent_image)
        
        print(f"Combining {len(patches)} patches: {', '.join(loaded_files)} with weights: {weights}")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight <= 0:
            print("Warning: Total weight is zero or negative. Using equal weights.")
            weights = [1.0 / len(patches)] * len(patches)
        else:
            weights = [w / total_weight for w in weights]
            
        # Initialize combined embeddings with the first patch weighted
        combined_image_prompt_embeds = patches[0]["image_prompt_embeds"] * weights[0]
        combined_uncond_image_prompt_embeds = patches[0]["uncond_image_prompt_embeds"] * weights[0]
        combined_face_kps = patches[0]["face_kps"] * weights[0]
        
        # Combine the other patches with their weights
        for i in range(1, len(patches)):
            combined_image_prompt_embeds += patches[i]["image_prompt_embeds"] * weights[i]
            combined_uncond_image_prompt_embeds += patches[i]["uncond_image_prompt_embeds"] * weights[i]
            combined_face_kps += patches[i]["face_kps"] * weights[i]
        
        # Set up the controlnet with the combined face keypoints
        cnets = {}
        cond_uncond = []
        
        # Process positive and negative conditioning separately
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                # Get the original weight and conditioning dictionary
                weight, cond_dict = t
                
                # Create a copy of the conditioning dictionary to avoid modifying the original
                d = cond_dict.copy()
                
                # Set up the control net
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(combined_face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                # Apply only the face-specific parts to the conditioning
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                
                # Add the face embeddings
                if is_cond:
                    d['cross_attn_controlnet'] = combined_image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                                  dtype=c_net.cond_hint_original.dtype)
                else:
                    d['cross_attn_controlnet'] = combined_uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                                  dtype=c_net.cond_hint_original.dtype)
                
                # Create the new conditioning entry with the original weight
                n = [weight, d]
                c.append(n)
            
            # Add the processed conditioning to the appropriate list
            cond_uncond.append(c)
            is_cond = False
        
        return(model, cond_uncond[0], cond_uncond[1], latent_image)


# Define NODE_CLASS_MAPPINGS after all classes are defined
NODE_CLASS_MAPPINGS = {
    "StarInfiniteYouPatch": StarInfiniteYouPatch,
    "StarInfiniteYouPatchCombine": StarInfiniteYouPatchCombine
}
