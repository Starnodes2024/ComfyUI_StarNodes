import os
import json
import torch
import folder_paths
import datetime
from pathlib import Path

# Define the background and title colors for the nodes
BGCOLOR = "#3d124d"  # Background color
COLOR = "#19124d"    # Title color

class StarConditioningSaver:
    """
    Node for saving conditioning data to a file for later use.
    """
    BGCOLOR = BGCOLOR
    COLOR = COLOR
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename": ("STRING", {"default": "conditioning"}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "save_conditioning"
    CATEGORY = "⭐StarNodes/Conditioning"
    
    def save_conditioning(self, conditioning, filename):
        # Create the conditionings directory if it doesn't exist
        output_dir = os.path.join(folder_paths.get_output_directory(), "conditionings")
        os.makedirs(output_dir, exist_ok=True)
        
        # Add timestamp to filename to make it unique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename:
            filename = "conditioning"
        
        # Sanitize filename
        filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
        full_filename = f"{filename}_{timestamp}"
        
        # Create the full path
        file_path = os.path.join(output_dir, f"{full_filename}.pt")
        
        # Save the conditioning data
        try:
            # Convert conditioning to a serializable format
            serializable_conditioning = []
            for cond_item in conditioning:
                cond, cond_data = cond_item
                
                # Convert tensors to lists for JSON serialization
                serialized_cond_data = {}
                for key, value in cond_data.items():
                    if isinstance(value, torch.Tensor):
                        serialized_cond_data[key] = value.cpu().detach().numpy().tolist()
                    else:
                        serialized_cond_data[key] = value
                
                serializable_conditioning.append([
                    cond.cpu().detach().numpy().tolist(),
                    serialized_cond_data
                ])
            
            # Save as torch file
            torch.save({
                "conditioning": conditioning,
                "metadata": {
                    "timestamp": timestamp,
                    "filename": filename
                }
            }, file_path)
            
            print(f"⭐ Star Conditioning saved to {file_path}")
        except Exception as e:
            print(f"Error saving conditioning: {str(e)}")
        
        # Return the original conditioning
        return (conditioning,)


class StarConditioningLoader:
    """
    Node for loading previously saved conditioning data.
    """
    BGCOLOR = BGCOLOR
    COLOR = COLOR
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_file": (cls.get_conditioning_files(),),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load_conditioning"
    CATEGORY = "⭐StarNodes/Conditioning"
    
    @classmethod
    def get_conditioning_files(cls):
        # Get the conditionings directory
        output_dir = os.path.join(folder_paths.get_output_directory(), "conditionings")
        os.makedirs(output_dir, exist_ok=True)
        
        # List all .pt files in the directory
        files = []
        for file in Path(output_dir).glob("*.pt"):
            files.append(file.name)
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        
        # Return empty list if no files found
        if not files:
            return ["No conditioning files found"]
        
        return files
    
    def load_conditioning(self, conditioning_file):
        # Check if the file exists
        if conditioning_file == "No conditioning files found":
            print("No conditioning files available to load")
            # Return empty conditioning
            return ([],)
        
        # Get the full path
        output_dir = os.path.join(folder_paths.get_output_directory(), "conditionings")
        file_path = os.path.join(output_dir, conditioning_file)
        
        try:
            # Load the conditioning data
            data = torch.load(file_path)
            conditioning = data["conditioning"]
            
            print(f"⭐ Star Conditioning loaded from {file_path}")
            return (conditioning,)
        except Exception as e:
            print(f"Error loading conditioning: {str(e)}")
            # Return empty conditioning
            return ([],)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StarConditioningSaver": StarConditioningSaver,
    "StarConditioningLoader": StarConditioningLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarConditioningSaver": "⭐ Star Conditioning Saver",
    "StarConditioningLoader": "⭐ Star Conditioning Loader",
}
