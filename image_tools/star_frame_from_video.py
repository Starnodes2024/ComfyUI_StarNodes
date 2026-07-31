import numpy as np
import torch
from typing import List, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarFrameFromVideo:
    """
    Node to select a single frame from a batch of images (e.g., from a video loader).
    User can choose First Frame, Last Frame, or a specific Frame Number.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),
                "frame_select_mode": (["First Frame", "Last Frame", "Frame Number"], {"default": "First Frame"}),
            },
            "optional": {
                "frame_number": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frame",)
    FUNCTION = "pick_frame"
    CATEGORY = "⭐StarNodes/Video"

    def pick_frame(self, images, frame_select_mode: str, frame_number: int = 0):
        # Log input type and shape for debugging
        if isinstance(images, torch.Tensor):
            logger.info(f"Input is torch.Tensor with shape {images.shape} and dtype {images.dtype}")
            # Get batch size from first dimension
            n = images.shape[0]
        elif isinstance(images, np.ndarray):
            logger.info(f"Input is numpy.ndarray with shape {images.shape} and dtype {images.dtype}")
            # Get batch size from first dimension
            n = images.shape[0]
        elif isinstance(images, list):
            logger.info(f"Input is list with length {len(images)}")
            n = len(images)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}. Expected torch.Tensor, numpy.ndarray, or list.")
            
        if n == 0:
            raise ValueError("Image batch is empty.")
            
        # Select frame index based on mode
        if frame_select_mode == "First Frame":
            idx = 0
        elif frame_select_mode == "Last Frame":
            idx = n - 1
        elif frame_select_mode == "Frame Number":
            idx = max(0, min(frame_number, n - 1))
        else:
            idx = 0
            
        logger.info(f"Selected frame index: {idx} out of {n} frames")
        
        # Simply select the frame without any conversion
        if isinstance(images, torch.Tensor):
            # For tensor, select the frame and keep the batch dimension
            selected_frame = images[idx:idx+1]
            logger.info(f"Selected tensor frame with shape: {selected_frame.shape}")
        elif isinstance(images, np.ndarray):
            # For numpy array, select the frame and keep the batch dimension
            selected_frame = images[idx:idx+1]
            logger.info(f"Selected numpy frame with shape: {selected_frame.shape}")
        else:  # list
            # For list, select the item
            selected_frame = [images[idx]]
            logger.info(f"Selected frame from list")
        
        return (selected_frame,)

NODE_CLASS_MAPPINGS = {"StarFrameFromVideo": StarFrameFromVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"StarFrameFromVideo": "⭐ Star Frame From Video"}
