import os
import torch
import numpy as np
from PIL import Image, ImageOps
import json
import hashlib
from pathlib import Path

class StarImageLoader1by1:
    """
    Star Image Loader 1by1 - Loads images from a folder one by one across workflow runs.
    
    Features:
    - Loads images sequentially from specified folder
    - Persists state between workflow runs using JSON file
    - Automatically resets counter when all images are processed
    - Supports common image formats
    - Provides progress information
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {"default": False}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "sort_by": (["name", "date", "size", "random"], {"default": "name"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "image_path", "current_index", "total_images", "remaining_images")
    FUNCTION = "load_next_image"
    CATEGORY = "⭐StarNodes/Image And Latent"
    
    def __init__(self):
        self.state_file = None
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
    
    def get_state_file_path(self, folder_path):
        """Generate state file path within the image source folder."""
        return os.path.join(folder_path, ".star_loader_state.json")
    
    def load_state(self, folder_path):
        """Load the current counter state from file."""
        state_file = self.get_state_file_path(folder_path)
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('counter', 0)
            except (json.JSONDecodeError, IOError):
                pass
        return 0
    
    def save_state(self, folder_path, counter):
        """Save the current counter state to file."""
        state_file = self.get_state_file_path(folder_path)
        try:
            with open(state_file, 'w') as f:
                json.dump({'counter': counter}, f)
        except IOError:
            # If we can't save state, it's not critical - just continue
            pass
    
    def get_image_files(self, folder, include_subfolders=False, sort_by="name", reverse_order=False):
        """Get all valid image files from the folder."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        image_files = []
        
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.valid_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in self.valid_extensions):
                    full_path = os.path.join(folder, file)
                    if os.path.isfile(full_path):
                        image_files.append(full_path)
        
        if not image_files:
            raise ValueError(f"No valid image files found in '{folder}'")
        
        # Sort files based on criteria
        if sort_by == "name":
            image_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=reverse_order)
        elif sort_by == "date":
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=reverse_order)
        elif sort_by == "size":
            image_files.sort(key=lambda x: os.path.getsize(x), reverse=reverse_order)
        elif sort_by == "random":
            import random
            random.shuffle(image_files)
        
        return image_files
    
    def load_image(self, image_path):
        """Load and preprocess an image."""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(img_array)[None,]
            
            # Create mask (all ones for RGB images)
            mask = torch.ones((64, 64), dtype=torch.float32, device="cpu")
            
            return img_tensor, mask
            
        except Exception as e:
            raise RuntimeError(f"Error loading image '{image_path}': {str(e)}")
    
    def load_next_image(self, folder, reset_counter=False, include_subfolders=False, sort_by="name", reverse_order=False):
        """Load the next image in sequence."""
        if not folder:
            raise ValueError("Folder path cannot be empty")
        
        # Get all image files
        image_files = self.get_image_files(folder, include_subfolders, sort_by, reverse_order)
        total_images = len(image_files)
        
        # Handle reset or first run
        if reset_counter:
            counter = 0
        else:
            counter = self.load_state(folder)
        
        # Ensure counter is within bounds
        if counter >= total_images:
            counter = 0  # Reset to start when all images are processed
        
        # Get the current image
        current_image_path = image_files[counter]
        
        # Load the image
        image, mask = self.load_image(current_image_path)
        
        # Increment counter for next run
        next_counter = counter + 1
        if next_counter >= total_images:
            next_counter = 0  # Reset to start over
        
        # Save the next counter state
        self.save_state(folder, next_counter)
        
        # Calculate remaining images
        remaining_images = total_images - counter - 1
        if remaining_images < 0:
            remaining_images = 0
        
        return (image, mask, current_image_path, counter, total_images, remaining_images)
    
    @classmethod
    def IS_CHANGED(s, folder, reset_counter=False, include_subfolders=False, sort_by="name", reverse_order=False):
        """
        This method determines when the node should re-execute.
        For this node, we want it to execute every time to get the next image.
        """
        # Always return a unique value to force re-execution
        import time
        return time.time()

# Node registration
NODE_CLASS_MAPPINGS = {
    "StarImageLoader1by1": StarImageLoader1by1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageLoader1by1": "⭐ Star Image Loader 1by1",
}
