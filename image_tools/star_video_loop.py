"""
Star Video Loop - Creates seamless looping frames from video inputs
Outputs frames that can be used with any Video Combine node.
Supports multiple video inputs that are joined horizontally to create sliding video panoramas.
"""

import numpy as np
import torch
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicVideoInputs(dict):
    """Flexible parameter definition for dynamic video inputs."""
    
    def __init__(self, base_dict=None):
        super().__init__(base_dict or {})
    
    def __getitem__(self, key):
        # Check if it's in the base dict first
        if key in self.keys():
            return super().__getitem__(key)
        # Dynamic video inputs: "video 2", "video 3", "video 4", etc. (with space)
        if key.startswith("video ") and key[6:].isdigit():
            return ("IMAGE", {"forceInput": True})  # Videos are IMAGE batches in ComfyUI
        raise KeyError(key)
    
    def __contains__(self, key):
        if super().__contains__(key):
            return True
        # Allow any video input pattern: "video N" with space
        if key.startswith("video ") and key[6:].isdigit():
            return True
        return False


class StarVideoLoop:
    """
    Creates seamless looping frames from video inputs.
    Videos are scrolled horizontally to create smooth, looping video frames.
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
        # Base optional inputs - first video slot is always visible
        base_optional = {
            "video 1": ("IMAGE", {"forceInput": True, "tooltip": "First video (IMAGE batch). Connect more videos to join them horizontally."}),
        }
        return {
            "required": {
                "resolution": (list(cls.RESOLUTIONS.keys()), {"default": "Full HD (1920)"}),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {"default": "1:1 (Square)"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.5}),
                "direction": (["Left to Right", "Right to Left"], {"default": "Left to Right"}),
            },
            "optional": DynamicVideoInputs(base_optional),
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "create_frames"
    CATEGORY = "⭐StarNodes/Video"
    DESCRIPTION = "Creates seamless looping video frames from video inputs. Connect multiple videos to join them horizontally into a sliding video panorama."

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

    def resize_frame(self, pil_image: Image.Image, target_height: int) -> Image.Image:
        """Resize a frame to target height while maintaining aspect ratio."""
        scale = target_height / pil_image.height
        new_width = int(pil_image.width * scale)
        return pil_image.resize((new_width, target_height), Image.Resampling.LANCZOS)

    def create_frames(self, resolution: str, aspect_ratio: str,
                      fps: int, duration: float, direction: str, **kwargs):
        """Create the video loop frames."""
        
        # Collect all connected videos (format: "video 1", "video 2", etc.)
        video_indices = []
        
        for key, value in kwargs.items():
            if key.startswith("video ") and value is not None:
                try:
                    idx = int(key.split(" ")[1])
                    video_indices.append((idx, value))
                except (ValueError, IndexError):
                    continue
        
        # Sort by index to maintain order
        video_indices.sort(key=lambda x: x[0])
        
        if not video_indices:
            raise ValueError("At least one video must be connected!")
        
        # Get resolution and aspect ratio values
        output_width = self.RESOLUTIONS[resolution]
        ratio_w, ratio_h = self.ASPECT_RATIOS[aspect_ratio]
        output_height = int(output_width * ratio_h / ratio_w)
        
        # Ensure dimensions are even (required for video encoding)
        output_width = output_width if output_width % 2 == 0 else output_width + 1
        output_height = output_height if output_height % 2 == 0 else output_height + 1
        
        logger.info(f"Creating video loop: {output_width}x{output_height} @ {fps}fps, {duration}s")
        
        # Process each video - get frame counts and resize
        videos_data = []
        for idx, tensor in video_indices:
            # tensor shape: [N, H, W, C] where N is frame count
            frame_count = tensor.shape[0]
            logger.info(f"Video {idx}: {frame_count} frames, {tensor.shape[2]}x{tensor.shape[1]}")
            videos_data.append({
                'idx': idx,
                'tensor': tensor,
                'frame_count': frame_count,
            })
        
        # Find the minimum frame count (we'll loop shorter videos)
        min_frames = min(v['frame_count'] for v in videos_data)
        
        # Calculate total output frames
        total_output_frames = int(fps * duration)
        
        # For each output frame, we need to:
        # 1. Determine which source frame to use (loop through source frames)
        # 2. Join all videos horizontally for that frame
        # 3. Scroll/crop to get the output frame
        
        # First, let's build the panorama width by checking first frame of each video
        panorama_widths = []
        for v in videos_data:
            first_frame = self.tensor_to_pil(v['tensor'][0])
            resized = self.resize_frame(first_frame, output_height)
            panorama_widths.append(resized.width)
            v['frame_width'] = resized.width
        
        total_panorama_width = sum(panorama_widths)
        logger.info(f"Total panorama width: {total_panorama_width}px from {len(videos_data)} videos")
        
        # Calculate scroll parameters
        scroll_distance = total_panorama_width  # One complete panorama for perfect loop
        pixels_per_frame = scroll_distance / total_output_frames
        
        logger.info(f"Output frames: {total_output_frames}, scroll: {scroll_distance}px, px/frame: {pixels_per_frame:.2f}")
        
        # Generate output frames
        frame_tensors = []
        
        for out_frame_idx in range(total_output_frames):
            # Calculate which source frame to use (loop through source frames)
            source_frame_idx = out_frame_idx % min_frames
            
            # Build the panorama for this frame by joining all videos
            x_offset = 0
            panorama = Image.new('RGB', (total_panorama_width * 2, output_height))  # 2x for seamless tiling
            
            # First pass - build the panorama
            for v in videos_data:
                # Get the frame (loop if needed)
                frame_idx = source_frame_idx % v['frame_count']
                frame_pil = self.tensor_to_pil(v['tensor'][frame_idx])
                frame_resized = self.resize_frame(frame_pil, output_height)
                
                # Paste twice for seamless tiling
                panorama.paste(frame_resized, (x_offset, 0))
                panorama.paste(frame_resized, (x_offset + total_panorama_width, 0))
                x_offset += frame_resized.width
            
            # Calculate scroll position
            if direction == "Left to Right":
                scroll_x = int(out_frame_idx * pixels_per_frame)
            else:  # Right to Left
                scroll_x = scroll_distance - int(out_frame_idx * pixels_per_frame)
            
            # Ensure scroll_x is within bounds
            scroll_x = scroll_x % total_panorama_width
            
            # Crop the output frame
            output_frame = panorama.crop((scroll_x, 0, scroll_x + output_width, output_height))
            frame_tensors.append(self.pil_to_tensor(output_frame))
            
            if out_frame_idx % 50 == 0:
                logger.info(f"Generated frame {out_frame_idx + 1}/{total_output_frames}")
        
        logger.info(f"Generated {len(frame_tensors)} frames")
        
        # Stack all frames into a batch tensor [N, H, W, C]
        frames_batch = torch.stack(frame_tensors, dim=0)
        
        return (frames_batch, fps)


NODE_CLASS_MAPPINGS = {
    "StarVideoLoop": StarVideoLoop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoLoop": "⭐ Star Video Loop"
}
