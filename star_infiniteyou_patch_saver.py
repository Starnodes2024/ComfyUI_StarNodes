import torch
import os
import folder_paths
import copy

# Ensure output directory exists for patch files
PATCH_DIR = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
os.makedirs(PATCH_DIR, exist_ok=True)

# Star InfiniteYou Patch Saver node implementation

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarInfiniteYouSaver": "⭐Star InfiniteYou Patch Saver"
}

class StarInfiniteYouSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "patch_data": ("PATCH_DATA",),
                "save_name": ("STRING", {"default": "my_patch"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_patch"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def save_patch(self, patch_data, save_name):
        # Ensure the filename has no invalid characters
        save_name = "".join(c for c in save_name if c.isalnum() or c in "_-").strip()
        if not save_name:
            save_name = "unnamed_patch"
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full path for the patch file
        patch_path = os.path.join(output_dir, f"{save_name}.iyou")
        
        # Create a deep copy of the patch data to avoid modifying the original
        processed_data = copy.deepcopy(patch_data)
        
        # Process the conditioning data to make it serializable
        if "processed_positive" in processed_data and "processed_negative" in processed_data:
            # Create serializable versions of the conditioning
            processed_data["processed_positive"] = self.make_serializable(processed_data["processed_positive"])
            processed_data["processed_negative"] = self.make_serializable(processed_data["processed_negative"])
        
        # Convert any tensors to CPU before saving
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                processed_data[key] = value.detach().cpu()
        
        # Save the patch data
        try:
            torch.save(processed_data, patch_path)
            print(f"InfiniteYou patch saved to: {patch_path}")
        except Exception as e:
            print(f"Error saving patch: {str(e)}")
            # Fallback to saving just the embeddings if full conditioning can't be saved
            try:
                fallback_data = {
                    "image_prompt_embeds": processed_data["image_prompt_embeds"].detach().cpu() if isinstance(processed_data["image_prompt_embeds"], torch.Tensor) else processed_data["image_prompt_embeds"],
                    "uncond_image_prompt_embeds": processed_data["uncond_image_prompt_embeds"].detach().cpu() if isinstance(processed_data["uncond_image_prompt_embeds"], torch.Tensor) else processed_data["uncond_image_prompt_embeds"],
                    "face_kps": processed_data["face_kps"].detach().cpu() if isinstance(processed_data["face_kps"], torch.Tensor) else processed_data["face_kps"],
                    "cn_strength": processed_data["cn_strength"],
                    "start_at": processed_data["start_at"],
                    "end_at": processed_data["end_at"]
                }
                torch.save(fallback_data, patch_path)
                print(f"Saved fallback patch data (without processed conditioning) to: {patch_path}")
            except Exception as e2:
                print(f"Failed to save even fallback data: {str(e2)}")
        
        return {}
    
    def make_serializable(self, conditioning):
        """Convert conditioning to a serializable format by removing non-serializable objects"""
        serializable_conditioning = []
        
        for cond in conditioning:
            # Each conditioning item is a tuple (weight, dict)
            weight, cond_dict = cond
            
            # Create a new dict with only serializable items
            serializable_dict = {}
            
            # Copy serializable items
            for k, v in cond_dict.items():
                # Skip known non-serializable keys
                if k in ['control']:
                    continue
                    
                # Try to keep tensors and basic types
                if isinstance(v, (int, float, str, bool, torch.Tensor)):
                    serializable_dict[k] = v
                elif v is None:
                    serializable_dict[k] = None
                # Skip complex objects
            
            # Add to the new conditioning list
            serializable_conditioning.append((weight, serializable_dict))
        
        return serializable_conditioning
