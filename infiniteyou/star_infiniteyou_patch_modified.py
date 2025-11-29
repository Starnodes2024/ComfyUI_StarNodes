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
    "StarInfiniteYouPatch": "⭐Star InfiniteYou Patch Loader",
    "StarInfiniteYouPatchCombine": "⭐Star InfiniteYou Patch Combine"
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
        
        import time
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "patch_file": (["none", "random"] + patches),
                # 'trigger' is a dummy input to break ComfyUI's node cache; connect a changing value (e.g., from Random Number node)
                "trigger": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_patch"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def apply_patch(self, control_net, model, positive, negative, vae, latent_image, patch_file, trigger):  # 'trigger' is unused except for cache-busting
        if patch_file == "none":
            # If no patch file is selected, just return the inputs unchanged
            return (model, positive, negative, latent_image)
        
        # Handle random patch selection
        if patch_file == "random":
            import random, time, torch
            patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
            patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
            if not patches:
                print("No patch files found in the patch directory.")
                return (model, positive, negative, latent_image)
            # Filter patches for only those with all required keys
            valid_patches = []
            print(f"[DEBUG] Found {len(patches)} patch files: {patches}")
            for pf in patches:
                patch_path = os.path.join(patch_dir, pf)
                try:
                    patch_data = torch.load(patch_path, map_location="cpu")
                    print(f"[DEBUG] Patch '{pf}' loaded. Keys: {list(patch_data.keys()) if hasattr(patch_data, 'keys') else type(patch_data)}")
                    if all(k in patch_data for k in ("image_prompt_embeds", "uncond_image_prompt_embeds", "face_kps")):
                        valid_patches.append(pf)
                except Exception as e:
                    print(f"[DEBUG] Skipping patch {pf} due to load error: {e}")
            print(f"[DEBUG] Valid patches after filtering: {valid_patches}")
            if not valid_patches:
                print("[DEBUG] No valid patch files (with all required keys) found in the patch directory.")
                return (model, positive, negative, latent_image)
            random.seed(time.time_ns())
            patch_file = random.choice(valid_patches)
            print(f"[DEBUG] Randomly selected valid patch file: {patch_file}")
        
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

        # Shape validation for face_kps
        import torch
        face_kps_tensor = torch.as_tensor(face_kps)
        print(f"Loaded face_kps from patch file: {patch_file}, shape: {face_kps_tensor.shape}")
        if face_kps_tensor.ndim != 4:
            print(f"Warning: face_kps in patch file {patch_file} does not have 4 dimensions (got {face_kps_tensor.ndim}). Skipping patch application.")
            return (model, positive, negative, latent_image)
        face_kps = face_kps_tensor

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
                "patch_file_1": (["none", "random"] + patches,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_2": (["none", "random"] + patches,),
                "weight_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "patch_file_3": (["none", "random"] + patches,),
                "weight_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_4": (["none", "random"] + patches,),
                "weight_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "patch_file_5": (["none", "random"] + patches,),
                "weight_5": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "PATCH_DATA")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent", "patch_data")
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
        
        # Function to handle random patch selection
        def get_random_patch():
            import random
            patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
            patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
            if not patches:
                print("No patch files found in the patch directory.")
                return "none"
            # Pick a random patch
            random_patch = random.choice(patches)
            print(f"Randomly selected patch file: {random_patch}")
            return random_patch
        
        # Process random selection for each patch slot
        if patch_file_1 == "random":
            patch_file_1 = get_random_patch()
        if patch_file_2 == "random":
            patch_file_2 = get_random_patch()
        if patch_file_3 == "random":
            patch_file_3 = get_random_patch()
        if patch_file_4 == "random":
            patch_file_4 = get_random_patch()
        if patch_file_5 == "random":
            patch_file_5 = get_random_patch()
        
        # Check if at least one patch file is selected
        if patch_file_1 == "none" and patch_file_2 == "none" and \
           (patch_file_3 is None or patch_file_3 == "none") and \
           (patch_file_4 is None or patch_file_4 == "none") and \
           (patch_file_5 is None or patch_file_5 == "none"):
            # If no patch file is selected, just return the inputs unchanged
            return (model, positive, negative, latent_image, None)
        
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
            return (model, positive, negative, latent_image, None)
        
        print(f"Combining {len(patches)} patches: {', '.join(loaded_files)} with weights: {weights}")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight <= 0:
            print("Warning: Total weight is zero or negative. Using equal weights.")
            weights = [1.0 / len(patches)] * len(patches)
        else:
            weights = [w / total_weight for w in weights]
            
        # Get shapes for reference
        print(f"Patch 0 face_kps shape: {patches[0]['face_kps'].shape}")
        
        # Check if all tensors have the same dimensions
        same_dimensions = True
        ref_shape = patches[0]['face_kps'].shape
        for i in range(1, len(patches)):
            if patches[i]['face_kps'].shape != ref_shape:
                same_dimensions = False
                print(f"Patch {i} has different shape: {patches[i]['face_kps'].shape} vs reference {ref_shape}")
                break
        
        if same_dimensions:
            print("All patches have the same dimensions. Simple combination.")
            # Initialize combined embeddings with the first patch weighted
            combined_image_prompt_embeds = patches[0]["image_prompt_embeds"] * weights[0]
            combined_uncond_image_prompt_embeds = patches[0]["uncond_image_prompt_embeds"] * weights[0]
            combined_face_kps = patches[0]["face_kps"] * weights[0]
            
            # Combine the other patches with their weights
            for i in range(1, len(patches)):
                combined_image_prompt_embeds += patches[i]["image_prompt_embeds"] * weights[i]
                combined_uncond_image_prompt_embeds += patches[i]["uncond_image_prompt_embeds"] * weights[i]
                combined_face_kps += patches[i]["face_kps"] * weights[i]
        else:
            # Handle different dimensions by using the dimensions of the first patch as reference
            print("Patches have different dimensions. Resizing all to match the first patch.")
            
            import torch.nn.functional as F
            
            # Initialize combined embeddings with the first patch weighted
            combined_image_prompt_embeds = patches[0]["image_prompt_embeds"] * weights[0]
            combined_uncond_image_prompt_embeds = patches[0]["uncond_image_prompt_embeds"] * weights[0]
            combined_face_kps = patches[0]["face_kps"] * weights[0]
            
            # Reference shapes from the first patch
            ref_face_kps_shape = patches[0]["face_kps"].shape
            
            # Combine the other patches with their weights, resizing if necessary
            for i in range(1, len(patches)):
                # No need to resize prompt embeddings - they should have the same dim for the same model
                combined_image_prompt_embeds += patches[i]["image_prompt_embeds"] * weights[i]
                combined_uncond_image_prompt_embeds += patches[i]["uncond_image_prompt_embeds"] * weights[i]
                
                # For face keypoints, check if resize is needed
                curr_face_kps = patches[i]["face_kps"]
                if curr_face_kps.shape != ref_face_kps_shape:
                    print(f"Resizing face keypoints for patch {i} from {curr_face_kps.shape} to {ref_face_kps_shape}")
                    
                    # Resize face keypoints - adapt based on dimensions
                    # Assuming face_kps is [B, H, W, C] format
                    B, H, W, C = ref_face_kps_shape
                    
                    # Move channel dim for interpolation
                    curr_face_kps = curr_face_kps.movedim(-1, 1)  # [B, C, H, W]
                    
                    # Resize height and width
                    resized_face_kps = F.interpolate(curr_face_kps, size=(H, W), mode='bilinear', align_corners=False)
                    
                    # Move channel back
                    resized_face_kps = resized_face_kps.movedim(1, -1)  # [B, H, W, C]
                    
                    # Add to combined with weight
                    combined_face_kps += resized_face_kps * weights[i]
                else:
                    combined_face_kps += curr_face_kps * weights[i]
        
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
        
        # Create patch data for saving
        patch_data = {
            "image_prompt_embeds": combined_image_prompt_embeds.cpu(),
            "uncond_image_prompt_embeds": combined_uncond_image_prompt_embeds.cpu(),
            "face_kps": combined_face_kps.cpu(),
            "cn_strength": cn_strength,
            "start_at": start_at,
            "end_at": end_at
        }
        
        return(model, cond_uncond[0], cond_uncond[1], latent_image, patch_data)


# Define NODE_CLASS_MAPPINGS after all classes are defined
NODE_CLASS_MAPPINGS = {
    "StarInfiniteYouPatch": StarInfiniteYouPatch,
    "StarInfiniteYouPatchCombine": StarInfiniteYouPatchCombine
}
