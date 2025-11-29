import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image

from .starinfiniteyou_resampler import Resampler
from .starinfiniteyou_utils import tensor_to_image, draw_kps, add_noise
from .starinfiniteyou_core import InfiniteYou

# Star InfiniteYou Apply node implementation - fixed to match original

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarApplyInfiniteYou": "⭐Star InfiniteYou Apply"
}

class StarApplyInfiniteYou:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "ref_image": ("IMAGE",),
                "latent_image": ("LATENT",),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "weight": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "vae": ("VAE",),
                "fixed_face_pose": ("BOOLEAN", {"default": False, "tooltip": "Fix the face pose from reference image."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "PATCH_DATA")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent", "patch_data")
    FUNCTION = "apply_infinite_you"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model

    def apply_infinite_you(self, adapter_file, control_net, ref_image, model, positive, negative, start_at, end_at, vae, latent_image, fixed_face_pose, weight=0.99):
        # Use constant values for these parameters that were originally hidden in the node
        noise = 0.35
        balance = 0.5
        cn_strength = weight  # Use weight as cn_strength like in the original
        
        ref_image = tensor_to_image(ref_image)
        ref_image = PIL.Image.fromarray(ref_image.astype(np.uint8))
        ref_image = ref_image.convert("RGB")

        # Load resampler model
        infiniteyou_model = self.load_model(adapter_file)
        
        # Get proper dtype for the model
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        device = comfy.model_management.get_torch_device()

        # Extract face embedding and landmarks
        face_emb, face_kps = infiniteyou_model.get_face_embed_and_landmark(ref_image)
        if face_emb is None:
            raise Exception('Reference Image: No face detected.')
            
        clip_embed = face_emb
        # Handle multiple embeddings if present (not likely in most cases)
        if clip_embed.shape[0] > 1:
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)

        # Add noise to embedding if specified
        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # Move model and tensors to the same device and dtype
        infiniteyou_model = infiniteyou_model.to(device, dtype=dtype)
        clip_embed = clip_embed.to(device, dtype=dtype)
        clip_embed_zeroed = clip_embed_zeroed.to(device, dtype=dtype)
        
        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)

        # Create face keypoints visualization for controlnet that matches target dimensions
        out = []
        height = latent_image['samples'].shape[2] * 8
        width = latent_image['samples'].shape[3] * 8
        
        if fixed_face_pose:
            # Use reference image size if fixed pose is enabled
            from .starinfiniteyou_utils import resize_and_pad_image
            control_image = resize_and_pad_image(ref_image, (width, height))
            image_kps = draw_kps(control_image, face_kps)
        else:
            # Create empty keypoints image if no fixed pose
            image_kps = np.zeros([height, width, 3])
            image_kps = PIL.Image.fromarray(image_kps.astype(np.uint8))
            
        # Convert to tensor using the same approach as original
        import torchvision.transforms as T
        out.append(image_kps)
        # Apply ToTensor to each element in the list then stack the results
        transform = T.ToTensor()
        tensor_list = [transform(img) for img in out]
        face_kps_tensor = torch.stack(tensor_list, dim=0).permute([0,2,3,1])
        # Move tensor to the correct device and dtype
        face_kps_tensor = face_kps_tensor.to(device, dtype=dtype)

        # Set up controlnet with face keypoints
        cnets = {}
        cond_uncond = []

        # Process positive and negative conditioning
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                weight, cond_dict = t
                d = cond_dict.copy()

                # Set up control network
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps_tensor.movedim(-1, 1), 
                                                            cn_strength, 
                                                            (start_at, end_at), 
                                                            vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                # Apply face-specific conditioning
                d['control'] = c_net
                d['control_apply_to_uncond'] = False

                # Add face embeddings
                if is_cond:
                    d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)
                else:
                    d['cross_attn_controlnet'] = uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)

                c.append([weight, d])
            
            cond_uncond.append(c)
            is_cond = False

        # Create patch data for saving
        patch_data = {
            "image_prompt_embeds": image_prompt_embeds.cpu(),
            "uncond_image_prompt_embeds": uncond_image_prompt_embeds.cpu(),
            "face_kps": face_kps_tensor.cpu(),
            "cn_strength": cn_strength,
            "start_at": start_at,
            "end_at": end_at
        }

        return (model, cond_uncond[0], cond_uncond[1], latent_image, patch_data)
