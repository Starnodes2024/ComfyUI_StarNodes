import os
import torch
import numpy as np
from PIL import Image
import time

class StarLTXVLoadLastImage:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_last_image"
    CATEGORY = "⭐StarNodes/LTX Video"
    
    @classmethod
    def IS_CHANGED(cls, folder_path):
        if not folder_path or not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return float("nan")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}
        image_files = []
        
        try:
            for filename in os.listdir(folder_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in image_extensions:
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        image_files.append(file_path)
            
            if image_files:
                newest_image = max(image_files, key=os.path.getmtime)
                mtime = os.path.getmtime(newest_image)
                return f"{newest_image}_{mtime}"
        except:
            pass
        
        return time.time()
    
    def load_last_image(self, folder_path):
        if not folder_path or not os.path.exists(folder_path):
            print(f"[StarLTXVLoadLastImage] Folder path does not exist: {folder_path}")
            return (self.create_default_image(),)
        
        if not os.path.isdir(folder_path):
            print(f"[StarLTXVLoadLastImage] Path is not a directory: {folder_path}")
            return (self.create_default_image(),)
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}
        
        image_files = []
        for filename in os.listdir(folder_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    image_files.append(file_path)
        
        if not image_files:
            print(f"[StarLTXVLoadLastImage] No image files found in folder: {folder_path}")
            return (self.create_default_image(),)
        
        newest_image = max(image_files, key=os.path.getmtime)
        
        print(f"[StarLTXVLoadLastImage] Loading newest image: {newest_image}")
        
        try:
            img = Image.open(newest_image)
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 256)).convert('L')
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img).astype(np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_array)
            
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
            
            img_tensor = img_tensor.unsqueeze(0)
            
            return (img_tensor,)
            
        except Exception as e:
            print(f"[StarLTXVLoadLastImage] Error loading image: {e}")
            return (self.create_default_image(),)
    
    def create_default_image(self):
        default_img = np.ones((512, 512, 3), dtype=np.float32) * 0.5
        img_tensor = torch.from_numpy(default_img).unsqueeze(0)
        return img_tensor


NODE_CLASS_MAPPINGS = {
    "StarLTXVLoadLastImage": StarLTXVLoadLastImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLTXVLoadLastImage": "⭐ Star LTXV Load Last Image From Folder",
}
