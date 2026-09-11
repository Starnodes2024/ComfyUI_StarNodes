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
                    "tooltip": "Path to source PNG/WEBP file with metadata to copy (e.g., 'output/ComfyUI_00001_.png')"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the output filename"
                }),
                "save_as": (["png", "webp"], {
                    "default": "png",
                    "tooltip": "Output format. PNG preserves standard ComfyUI PNG metadata. WEBP stores metadata in EXIF tags."
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

                # ComfyUI stores metadata in custom PNG chunks under the "comf" key
                # in the form: b"<name>\0<json>" (latin-1).
                comf_chunks = self._extract_png_comf_chunks(file_path)
                if comf_chunks:
                    metadata.update(comf_chunks)

                # WebP metadata (ComfyUI-style) is stored in EXIF tags as strings "key:<json>"
                try:
                    exif_data = img.getexif()
                    if exif_data is not None:
                        for _, exif_value in exif_data.items():
                            if isinstance(exif_value, str):
                                if exif_value.startswith("prompt:"):
                                    metadata["prompt"] = exif_value[len("prompt:"):]
                                elif ":" in exif_value:
                                    k, v = exif_value.split(":", 1)
                                    if k and v:
                                        metadata[k] = v
                except Exception:
                    pass
                
                print(f"⭐ Star Meta Injector: Extracted {len(metadata)} metadata fields from source")
                
                # Log the metadata keys found
                if metadata:
                    print(f"⭐ Star Meta Injector: Metadata keys: {', '.join(metadata.keys())}")
                
                return metadata
                
        except Exception as e:
            print(f"⭐ Star Meta Injector: Error extracting metadata: {str(e)}")
            return {}

    def _extract_png_comf_chunks(self, file_path):
        if not file_path.lower().endswith('.png'):
            return {}

        try:
            with open(file_path, 'rb') as f:
                sig = f.read(8)
                if sig != b"\x89PNG\r\n\x1a\n":
                    return {}

                out = {}
                while True:
                    length_bytes = f.read(4)
                    if len(length_bytes) != 4:
                        break
                    length = int.from_bytes(length_bytes, 'big')
                    chunk_type = f.read(4)
                    if len(chunk_type) != 4:
                        break
                    data = f.read(length)
                    f.read(4)  # crc

                    if chunk_type == b"comf" and data and b"\x00" in data:
                        name_bytes, payload_bytes = data.split(b"\x00", 1)
                        try:
                            name = name_bytes.decode('latin-1', 'strict')
                            payload = payload_bytes.decode('latin-1', 'strict')
                            out[name] = payload
                        except Exception:
                            pass

                    if chunk_type == b"IEND":
                        break

                return out
        except Exception:
            return {}
    
    def inject_and_save(self, target_image, source_image_path, filename_prefix="ComfyUI", save_as="png"):
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

        # Create WebP EXIF metadata (ComfyUI-style: prompt in 0x0110, extra fields in 0x010F downward)
        webp_exif = None
        if save_as == "webp":
            try:
                webp_exif = target_pil.getexif()
                if "prompt" in metadata and isinstance(metadata.get("prompt"), str):
                    webp_exif[0x0110] = "prompt:{}".format(metadata.get("prompt"))

                inital_exif_tag = 0x010F
                for key, value in metadata.items():
                    if key == "prompt":
                        continue
                    if not isinstance(value, str):
                        value = str(value)
                    webp_exif[inital_exif_tag] = "{}:{}".format(key, value)
                    inital_exif_tag -= 1
            except Exception as e:
                print(f"⭐ Star Meta Injector: Error creating WEBP EXIF metadata: {str(e)}")
                webp_exif = None
        
        # Generate unique filename
        counter = 1
        while True:
            ext = "png" if save_as == "png" else "webp"
            filename = f"{filename_prefix}_{counter:05d}_.{ext}"
            save_path = os.path.join(self.output_dir, filename)
            if not os.path.exists(save_path):
                break
            counter += 1
        
        # Save with metadata
        if save_as == "webp":
            target_pil.save(save_path, exif=webp_exif, lossless=True, quality=100, method=6)
        else:
            # Also embed ComfyUI-style chunks for prompt/workflow compatibility where possible
            try:
                if "prompt" in metadata and isinstance(metadata.get("prompt"), str):
                    png_info.add(b"comf", "prompt".encode("latin-1", "strict") + b"\0" + metadata.get("prompt").encode("latin-1", "strict"), after_idat=True)
                for key, value in metadata.items():
                    if key == "prompt":
                        continue
                    if not isinstance(value, str):
                        value = str(value)
                    png_info.add(b"comf", key.encode("latin-1", "strict") + b"\0" + value.encode("latin-1", "strict"), after_idat=True)
            except Exception:
                pass
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
