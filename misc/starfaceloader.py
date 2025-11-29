import os
import folder_paths
import shutil
from PIL import Image
import torch
import numpy as np
from nodes import PreviewImage, SaveImage
import time

class FlexibleInputs(dict):
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    """A special class to make flexible node inputs."""
    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type, )

    def __contains__(self, key):
        return True

class StarFaceLoader:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if "image" in kwargs:
            return True
        return False
        
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        faces_dir = os.path.join(input_dir, "faces")
        files = []
        
        # If faces directory exists, show images from there
        if os.path.exists(faces_dir):
            for f in os.listdir(faces_dir):
                if os.path.isfile(os.path.join(faces_dir, f)):
                    files.append(os.path.join("faces", f))
        else:
            # If no faces directory, show all images from input directory
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)
                    
        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True,
                    "__type__": "STRING",
                }),
                "upload_to_face_folder": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_face_image"
    OUTPUT_NODE = True

    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        self.faces_dir = os.path.join(self.input_dir, "faces")
        self.pasted_dir = os.path.join(self.input_dir, "pasted")
        # Create necessary directories
        for d in [self.faces_dir, self.pasted_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def generate_timestamped_filename(self, original_filename):
        """Generate a filename with timestamp to avoid conflicts"""
        name, ext = os.path.splitext(original_filename)
        timestamp = int(time.time())
        return f"{name}_{timestamp}{ext}"

    def save_image(self, image_path, save_path):
        """Helper to save image"""
        i = Image.open(image_path)
        i = i.convert('RGB')
        i.save(save_path, format='PNG')
        
        # Convert to tensor
        image = np.array(i).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        i.close()
        return image
            
    def load_face_image(self, image, upload_to_face_folder):
        # Get the annotated file path (handles both regular and pasted images)
        image_path = folder_paths.get_annotated_filepath(image)
        original_filename = os.path.basename(image)
        
        # Only process new files if it's a pasted image (temp) or a new upload
        is_new_upload = not image.startswith("faces/") and "temp" not in image_path and not os.path.exists(os.path.join(self.faces_dir, original_filename))
        if "temp" in image_path or is_new_upload:
            try:
                # Generate timestamped filename only for pasted images
                if "temp" in image_path:
                    filename = self.generate_timestamped_filename(original_filename)
                else:
                    # For uploads, use the original filename
                    filename = original_filename
                
                if "temp" in image_path:
                    source_path = image_path
                else:
                    source_path = os.path.join(self.input_dir, image)
                
                # Always save to input directory for new files
                input_path = os.path.join(self.input_dir, filename)
                if not os.path.exists(input_path):
                    image_tensor = self.save_image(source_path, input_path)
                
                # If it's a pasted image, also save to pasted directory
                if "temp" in image_path:
                    pasted_path = os.path.join(self.pasted_dir, filename)
                    self.save_image(source_path, pasted_path)
                
                # If upload_to_face_folder is true, save to faces with same filename
                if upload_to_face_folder:
                    face_path = os.path.join(self.faces_dir, filename)
                    if not os.path.exists(face_path):
                        self.save_image(source_path, face_path)
                    # Use faces path for loading
                    image = os.path.join("faces", filename)
                else:
                    # Use input path
                    image = filename
                    
            except Exception as e:
                print(f"Error handling new image: {e}")
                raise

        # Always construct final path from input directory
        image_path = os.path.join(self.input_dir, image)

        # Open and process the image
        i = Image.open(image_path)
        i = i.convert('RGB')
        image = i.copy()
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        # Create an empty mask of the same size
        mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        
        # Get preview results
        preview_basename = os.path.splitext(original_filename)[0]
        preview_results = SaveImage().save_images(image, preview_basename, None, None)

        return {"ui": {"images": preview_results['ui']['images']},
                "result": (image, mask)}

NODE_CLASS_MAPPINGS = {
    "Star Face Loader": StarFaceLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Star Face Loader": "⭐ Star Face Loader (img)"
}
