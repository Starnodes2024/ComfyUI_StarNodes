"""
Star Image Loop - Creates seamless looping frames from panoramic images
Outputs frames that can be used with any Video Combine node.
Supports multiple image inputs that are joined horizontally to create longer panoramas.
"""

import numpy as np
import torch
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicImageInputs(dict):
    """Flexible parameter definition for dynamic image inputs."""
    
    def __init__(self, base_dict=None):
        super().__init__(base_dict or {})
    
    def __getitem__(self, key):
        # Check if it's in the base dict first
        if key in self.keys():
            return super().__getitem__(key)
        # Dynamic image inputs: "image 2", "image 3", "image 4", etc. (with space)
        if key.startswith("image ") and key[6:].isdigit():
            return ("IMAGE", {"forceInput": True})
        raise KeyError(key)
    
    def __contains__(self, key):
        if super().__contains__(key):
            return True
        # Allow any image input pattern: "image N" with space
        if key.startswith("image ") and key[6:].isdigit():
            return True
        return False


class StarImageLoop:
    """
    Creates seamless looping frames from panoramic images.
    Images are scrolled horizontally to create smooth, looping video frames.
    Connect the output to any Video Combine node to create the final video.
    """
    
    # Resolution presets (width values)
    RESOLUTIONS = {
        "HD (1280)": 1280,
        "Full HD (1920)": 1920,
        "2K (2560)": 2560,
        "4K (3840)": 3840,
    }
    
    # Aspect ratio presets (width:height)
    ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "9:16 (TikTok/Reels)": (9, 16),
        "4:5 (Instagram)": (4, 5),
        "16:9 (YouTube)": (16, 9),
        "3:4 (Portrait)": (3, 4),
        "2:3 (Portrait)": (2, 3),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Base optional inputs - first image slot is always visible
        base_optional = {
            "image 1": ("IMAGE", {"forceInput": True, "tooltip": "First panorama image. Connect more images to join them horizontally."}),
        }
        return {
            "required": {
                "resolution": (list(cls.RESOLUTIONS.keys()), {"default": "Full HD (1920)"}),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {"default": "1:1 (Square)"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.5}),
                "direction": (["Left to Right", "Right to Left"], {"default": "Left to Right"}),
            },
            "optional": DynamicImageInputs(base_optional),
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "create_frames"
    CATEGORY = "⭐StarNodes/Video"
    DESCRIPTION = "Creates seamless looping video frames from panoramic images. Connect multiple images to join them horizontally into a longer panorama."

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor to PIL Image."""
        # tensor shape: [H, W, C] or [1, H, W, C]
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(np_image)

    def create_seamless_panorama(self, pil_image: Image.Image, target_width: int):
        """
        Create a seamless tileable panorama by duplicating the image.
        The image is tiled enough times to cover at least 2x the target width
        to ensure smooth looping.
        """
        img_width, img_height = pil_image.size
        
        # Calculate how many times we need to tile the image
        # We need at least 2x target_width for seamless looping
        tiles_needed = max(2, int(np.ceil((target_width * 2) / img_width)) + 1)
        
        # Create the tiled image
        tiled_width = img_width * tiles_needed
        tiled_image = Image.new('RGB', (tiled_width, img_height))
        
        for i in range(tiles_needed):
            tiled_image.paste(pil_image, (i * img_width, 0))
        
        return tiled_image, img_width

    def join_images_horizontally(self, images: list, target_height: int) -> Image.Image:
        """
        Join multiple PIL images horizontally, scaling them to match target_height.
        """
        if not images:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            # Single image - just resize to target height
            img = images[0]
            scale = target_height / img.height
            new_width = int(img.width * scale)
            return img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        
        # Scale all images to target height and calculate total width
        scaled_images = []
        total_width = 0
        
        for img in images:
            scale = target_height / img.height
            new_width = int(img.width * scale)
            scaled_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            scaled_images.append(scaled_img)
            total_width += new_width
        
        # Create combined image
        combined = Image.new('RGB', (total_width, target_height))
        x_offset = 0
        
        for img in scaled_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        logger.info(f"Joined {len(images)} images into panorama: {total_width}x{target_height}")
        return combined

    def create_frames(self, resolution: str, aspect_ratio: str,
                      fps: int, duration: float, direction: str, **kwargs):
        """Create the panorama slide frames."""
        
        # Collect all connected images (format: "image 1", "image 2", etc.)
        image_indices = []
        
        for key, value in kwargs.items():
            if key.startswith("image ") and value is not None:
                try:
                    idx = int(key.split(" ")[1])
                    image_indices.append((idx, value))
                except (ValueError, IndexError):
                    continue
        
        # Sort by index to maintain order
        image_indices.sort(key=lambda x: x[0])
        
        if not image_indices:
            raise ValueError("At least one image must be connected!")
        
        # Convert tensors to PIL images
        pil_images = []
        for idx, tensor in image_indices:
            pil_img = self.tensor_to_pil(tensor)
            pil_images.append(pil_img)
            logger.info(f"Image {idx}: {pil_img.width}x{pil_img.height}")
        
        # Get resolution and aspect ratio values
        output_width = self.RESOLUTIONS[resolution]
        ratio_w, ratio_h = self.ASPECT_RATIOS[aspect_ratio]
        output_height = int(output_width * ratio_h / ratio_w)
        
        # Ensure dimensions are even (required for video encoding)
        output_width = output_width if output_width % 2 == 0 else output_width + 1
        output_height = output_height if output_height % 2 == 0 else output_height + 1
        
        logger.info(f"Creating frames: {output_width}x{output_height} @ {fps}fps, {duration}s")
        
        # Join all images horizontally at output height (already scaled)
        pil_image = self.join_images_horizontally(pil_images, output_height)
        panorama_width = pil_image.size[0]
        
        logger.info(f"Combined panorama size: {panorama_width}x{output_height}")
        
        # Create seamless tiled panorama
        tiled_image, single_tile_width = self.create_seamless_panorama(pil_image, output_width)
        tiled_width = tiled_image.size[0]
        logger.info(f"Tiled panorama size: {tiled_width}x{output_height}")
        
        # Calculate total frames and scroll distance
        total_frames = int(fps * duration)
        
        # For seamless loop, we scroll exactly one panorama width over the duration
        scroll_distance = panorama_width  # One complete panorama for perfect loop
        pixels_per_frame = scroll_distance / total_frames
        
        logger.info(f"Total frames: {total_frames}, scroll distance: {scroll_distance}px, pixels/frame: {pixels_per_frame:.2f}")
        
        # Generate frames
        frame_tensors = []
        
        for frame_idx in range(total_frames):
            # Calculate scroll position
            if direction == "Left to Right":
                scroll_x = int(frame_idx * pixels_per_frame)
            else:  # Right to Left
                scroll_x = scroll_distance - int(frame_idx * pixels_per_frame)
            
            # Ensure scroll_x is within bounds
            scroll_x = scroll_x % panorama_width
            
            # Crop the frame from the tiled image
            frame = tiled_image.crop((scroll_x, 0, scroll_x + output_width, output_height))
            frame_tensors.append(self.pil_to_tensor(frame))
        
        logger.info(f"Generated {len(frame_tensors)} frames")
        
        # Stack all frames into a batch tensor [N, H, W, C]
        frames_batch = torch.stack(frame_tensors, dim=0)
        
        return (frames_batch, fps)


NODE_CLASS_MAPPINGS = {
    "StarImageLoop": StarImageLoop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageLoop": "⭐ Star Image Loop"
}
