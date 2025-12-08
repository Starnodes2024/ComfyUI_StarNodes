"""
Star Meta Injector Node for ComfyUI
Transfers all metadata (including workflow data) from a source image to a target image and saves it.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class StarMetaInjector:
    """
    Transfers all PNG metadata (including ComfyUI workflow data) from a source image to a target image.
    Saves the target image with the source's metadata to preserve workflow information.
    
    This node allows you to:
    - Extract metadata from a source PNG image with embedded workflow data
    - Inject that metadata into a different target image
    - Save the result directly to preserve the metadata
    - Share workflows with custom images easily
    
    The metadata includes:
    - ComfyUI workflow JSON
    - Prompt information
    - Generation parameters
    - Any other PNG text chunks
    """
    
    def __init__(self):
        try:
            import folder_paths
            self.output_dir = folder_paths.get_output_directory()
        except:
            self.output_dir = "output"
            os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE", {
                    "tooltip": "The image that will receive the metadata and be saved"
                }),
                "source_image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to source PNG file with metadata to copy (e.g., 'output/ComfyUI_00001_.png')"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the output filename"
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "inject_and_save"
    CATEGORY = "⭐StarNodes/Image And Latent"
    OUTPUT_NODE = True
    DESCRIPTION = "Transfers metadata from source image to target image and saves it with embedded workflow data"
    
    def extract_metadata_from_file(self, file_path):
        """
        Extract all PNG metadata from a file.
        
        Args:
            file_path: Path to PNG file
            
        Returns:
            Dictionary containing all PNG text chunks
        """
        if not os.path.exists(file_path):
            print(f"⭐ Star Meta Injector: Source file not found: {file_path}")
            return {}
        
        try:
            with Image.open(file_path) as img:
                metadata = {}
                
                # Extract all PNG text chunks
                if hasattr(img, 'text'):
                    metadata = dict(img.text)
                
                # Also check for info attribute (alternative metadata storage)
                if hasattr(img, 'info'):
                    for key, value in img.info.items():
                        if isinstance(value, (str, bytes)):
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8')
                                except:
                                    continue
                            metadata[key] = value
                
                print(f"⭐ Star Meta Injector: Extracted {len(metadata)} metadata fields from source")
                
                # Log the metadata keys found
                if metadata:
                    print(f"⭐ Star Meta Injector: Metadata keys: {', '.join(metadata.keys())}")
                
                return metadata
                
        except Exception as e:
            print(f"⭐ Star Meta Injector: Error extracting metadata: {str(e)}")
            return {}
    
    def inject_and_save(self, target_image, source_image_path, filename_prefix="ComfyUI"):
        """
        Inject metadata from source to target image and save it.
        
        Args:
            target_image: Target image tensor
            source_image_path: Path to source PNG file with metadata
            filename_prefix: Prefix for the output filename
            
        Returns:
            Dictionary with UI information for ComfyUI
        """
        # Extract metadata from source file
        metadata = {}
        if source_image_path and source_image_path.strip():
            metadata = self.extract_metadata_from_file(source_image_path.strip())
        
        if not metadata:
            print("⭐ Star Meta Injector: Warning - No metadata found to inject. Saving without metadata.")
        
        # Convert target tensor to PIL
        img_array = target_image[0].cpu().numpy()
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        target_pil = Image.fromarray(img_array, mode='RGB')
        
        # Create PngInfo and add metadata
        png_info = PngInfo()
        for key, value in metadata.items():
            if not isinstance(value, str):
                value = str(value)
            png_info.add_text(key, value)
        
        # Generate unique filename
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}_.png"
            save_path = os.path.join(self.output_dir, filename)
            if not os.path.exists(save_path):
                break
            counter += 1
        
        # Save with metadata
        target_pil.save(save_path, pnginfo=png_info, compress_level=4)
        
        print(f"⭐ Star Meta Injector: Saved image with {len(metadata)} metadata fields to: {save_path}")
        
        # Return UI information for ComfyUI
        return {
            "ui": {
                "images": [{
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                }]
            }
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "StarMetaInjector": StarMetaInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMetaInjector": "⭐ Star Meta Injector",
}
