import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import PIL.Image
import cv2
import copy
import math
import torchvision.transforms as transforms

from .starinfiniteyou_utils import tensor_to_image, add_noise, draw_kps
from .starinfiniteyou_core import InfiniteYou

# Ensure output directory exists for patch files
PATCH_DIR = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
os.makedirs(PATCH_DIR, exist_ok=True)

# Star InfiniteYou Advanced Patch Maker node implementation
class StarInfiniteYouAdvancedPatchMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "image_1": ("IMAGE",),
                "save_name": ("STRING", {"default": "blended_face_patch"}),
                "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_5": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("patch_path",)
    FUNCTION = "create_advanced_patch"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")
        model = InfiniteYou(adapter_model_state_dict)
        # Store the model's weight dtype for later use
        self.model_dtype = next(model.image_proj_model.parameters()).dtype
        return model
    
    def process_image(self, image_tensor):
        """Convert tensor image to PIL image for face detection"""
        image_np = tensor_to_image(image_tensor)
        return PIL.Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
    
    def create_advanced_patch(self, adapter_file, image_1, save_name, noise, cn_strength, start_at, end_at,
                             image_2=None, image_3=None, image_4=None, image_5=None,
                             weight_1=1.0, weight_2=0.5, weight_3=0.5, weight_4=0.5, weight_5=0.5):
        # Load the InfiniteYou model
        infiniteyou_model = self.load_model(adapter_file)
        
        # Process the first (required) image
        pil_image_1 = self.process_image(image_1)
        
        # Get face embedding and landmarks from the first image
        try:
            face_emb_1, face_kps_1 = infiniteyou_model.get_face_embed_and_landmark(pil_image_1)
        except Exception as e:
            print(f"Error extracting face from image 1: {str(e)}")
            return ("Error: Could not detect face in the first image",)
        
        # Initialize lists to store all valid embeddings and keypoints
        valid_embeddings = [face_emb_1]
        valid_keypoints = [face_kps_1]
        valid_weights = [weight_1]
        
        # Process optional images if provided
        optional_images = [(image_2, weight_2), (image_3, weight_3), (image_4, weight_4), (image_5, weight_5)]
        
        for idx, (img_tensor, weight) in enumerate(optional_images):
            if img_tensor is not None:
                try:
                    pil_img = self.process_image(img_tensor)
                    face_emb, face_kps = infiniteyou_model.get_face_embed_and_landmark(pil_img)
                    
                    # Add noise to embedding if specified
                    if noise > 0:
                        face_emb = add_noise(face_emb, noise)
                    
                    valid_embeddings.append(face_emb)
                    valid_keypoints.append(face_kps)
                    valid_weights.append(weight)
                    print(f"Successfully processed face from image {idx+2}")
                except Exception as e:
                    print(f"Warning: Could not process image {idx+2}: {str(e)}")
        
        # Normalize weights
        total_weight = sum(valid_weights)
        if total_weight == 0:
            normalized_weights = [1.0] + [0.0] * (len(valid_weights) - 1)
        else:
            normalized_weights = [w / total_weight for w in valid_weights]
        
        print(f"Using {len(valid_embeddings)} faces with normalized weights: {normalized_weights}")
        
        # Blend embeddings and keypoints based on normalized weights
        blended_emb = torch.zeros_like(valid_embeddings[0])
        for emb, weight in zip(valid_embeddings, normalized_weights):
            blended_emb += emb * weight
        
        # Normalize the blended embedding to unit length (important for face embeddings)
        blended_emb = torch.nn.functional.normalize(blended_emb, p=2, dim=0)
        
        # Check if the embedding is already batched (has batch dimension)
        if len(blended_emb.shape) == 1:
            # Add batch dimension if it's missing
            blended_emb = blended_emb.unsqueeze(0)
        
        # For keypoints, we need to be careful with the shape
        # Get the shape from the first keypoints tensor/array
        ref_kps_shape = valid_keypoints[0].shape
        
        # Check if keypoints are numpy arrays or torch tensors
        is_numpy = isinstance(valid_keypoints[0], np.ndarray)
        
        # Initialize blended keypoints with zeros
        if is_numpy:
            blended_kps = np.zeros_like(valid_keypoints[0], dtype=np.float32)
        else:
            blended_kps = torch.zeros_like(valid_keypoints[0])
        
        # Blend keypoints
        for kps, weight in zip(valid_keypoints, normalized_weights):
            # Ensure keypoints have the same shape
            if kps.shape != ref_kps_shape:
                # Handle numpy arrays differently than torch tensors
                if is_numpy:
                    # Resize numpy array
                    kps = cv2.resize(kps, (ref_kps_shape[1], ref_kps_shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    # Resize torch tensor
                    kps = torch.nn.functional.interpolate(
                        kps.unsqueeze(0).permute(0, 3, 1, 2),  # [1, C, H, W]
                        size=(ref_kps_shape[0], ref_kps_shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1).squeeze(0)  # [H, W, C]
            
            blended_kps += kps * weight
        
        # Convert to torch tensor if it's a numpy array
        if is_numpy:
            blended_kps = torch.from_numpy(blended_kps)
        
        # Create device tensors
        device = comfy.model_management.get_torch_device()
        
        # Use the model's actual dtype instead of the unet_dtype
        # This ensures we match the exact precision of the model weights
        clip_embed = blended_emb.to(device=device, dtype=self.model_dtype)
        clip_embed_zeroed = torch.zeros_like(clip_embed)
        
        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)
        
        # Create a visualization of the face keypoints for the ControlNet
        # First, create a blank PIL image with the right dimensions
        # We'll use the first image as reference for size
        ref_pil_img = self.process_image(image_1)
        
        # Generate keypoints visualization
        face_kps_image = draw_kps(ref_pil_img, blended_kps.cpu().numpy())
        
        # Convert to tensor in the format expected by ControlNet
        transform = transforms.ToTensor()
        face_kps_tensor = transform(face_kps_image).unsqueeze(0).permute(0, 2, 3, 1)
        
        # Create patch data in the same format as the Apply node
        patch_data = {
            "image_prompt_embeds": image_prompt_embeds.cpu(),
            "uncond_image_prompt_embeds": uncond_image_prompt_embeds.cpu(),
            "face_kps": face_kps_tensor.cpu(),  # Use the visualization tensor
            "cn_strength": cn_strength,
            "start_at": start_at,
            "end_at": end_at,
            "original_kps": blended_kps.cpu()  # Store original keypoints as backup
        }
        
        # Ensure the filename has no invalid characters
        save_name = "".join(c for c in save_name if c.isalnum() or c in "_-").strip()
        if not save_name:
            save_name = "blended_face_patch"
        
        # Create the full path for the patch file
        patch_path = os.path.join(PATCH_DIR, f"{save_name}.iyou")
        
        # Save the patch data
        try:
            torch.save(patch_data, patch_path)
            print(f"InfiniteYou advanced patch saved to: {patch_path}")
            return (f"Saved: {patch_path}",)
        except Exception as e:
            print(f"Error saving patch: {str(e)}")
            return (f"Error: {str(e)}",)

# Define node mappings
NODE_CLASS_MAPPINGS = {
    "StarInfiniteYouAdvancedPatchMaker": StarInfiniteYouAdvancedPatchMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarInfiniteYouAdvancedPatchMaker": "⭐Star InfiniteYou Advanced Patch Maker"
}
