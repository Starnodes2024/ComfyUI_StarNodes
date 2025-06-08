import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import PIL.Image
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import node_helpers
from .starinfiniteyou_utils import tensor_to_image, draw_kps, add_noise
from .starinfiniteyou_core import InfiniteYou

class StarInfiniteYouFaceSwapMod:
    @classmethod
    def INPUT_TYPES(s):
        # Get list of patch files
        patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(patch_dir, exist_ok=True)
        
        # Get both .pt and .iyou patch files (for compatibility)
        patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
        # Sort patches alphabetically for better organization
        patches.sort()
        
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "clip": ("CLIP",),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "vae": ("VAE",),
                "weight": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_kernel": ("INT", {"default": 9, "min": 3, "max": 31, "step": 2}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "patch_file": (["none", "random"] + patches,),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask": ("MASK",),
                "combine_embeds": (["average", "add", "subtract", "multiply"], {"default": "average"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_face_swap"
    CATEGORY = "⭐StarNodes/InfiniteYou"

    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")
        model = InfiniteYou(adapter_model_state_dict)
        return model

    def encode_prompt(self, clip, text):
        # Encode text prompt with CLIP
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def prepare_condition_inpainting(self, positive, negative, pixels, vae, mask):
        # Make sure dimensions are divisible by 8 (VAE requirement)
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        # Resize mask to match pixel dimensions
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(pixels.shape[1], pixels.shape[2]), 
            mode="bilinear"
        )
        
        # Clone pixels to avoid modifying the original
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # Crop to ensure dimensions are divisible by 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]
        
        # Apply mask to pixels
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Move all tensors to the same device
        pixels = pixels.to(device)
        mask = mask.to(device)
        m = (1.0 - mask.round()).squeeze(1).to(device, dtype=dtype)
        
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        
        # Encode with VAE
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)
        
        # Prepare output latent
        out_latent = {}
        out_latent["samples"] = orig_latent
        out_latent["noise_mask"] = mask
        
        # Modify conditioning
        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(
                conditioning, 
                {
                    "concat_latent_image": concat_latent,
                    "concat_mask": mask
                }
            )
            out.append(c)
        
        return out[0], out[1], out_latent

    def prepare_mask_and_landmark(self, model, image, blur_kernel):
        # Convert tensor to cv2 image
        image_np = tensor_to_image(image)
        image_cv2 = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Detect face and get landmarks
        try:
            face = model.detect_face(image_cv2)
            kps = face.kps
            
            # Create a simple oval face mask around the landmarks
            h, w = image_np.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Calculate center and radius of face oval from landmarks
            landmarks = np.array(kps)
            center_x = np.mean(landmarks[:, 0])
            center_y = np.mean(landmarks[:, 1])
            
            # Calculate radii based on face size
            radius_x = np.max(np.abs(landmarks[:, 0] - center_x)) * 1.6
            radius_y = np.max(np.abs(landmarks[:, 1] - center_y)) * 1.8
            
            # Create coordinate grid
            y, x = np.ogrid[:h, :w]
            
            # Create oval mask using the ellipse equation: (x-h)²/a² + (y-k)²/b² <= 1
            mask_array = ((x - center_x)**2 / (radius_x**2) + (y - center_y)**2 / (radius_y**2)) <= 1.0
            mask[mask_array] = 1.0
            
            # Apply Gaussian blur to smooth the mask edges
            if blur_kernel > 0:
                # Ensure blur_kernel is odd
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
            
            # Normalize mask to [0,1]
            mask = np.clip(mask, 0, 1)
            
            # Convert to tensor format
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            return mask_tensor, kps
        
        except Exception as e:
            print(f"Face detection failed: {e}")
            return None, None

    def apply_face_swap(self, control_net, model, image, clip, start_at, end_at, vae, adapter_file, weight=0.99, blur_kernel=9, cn_strength=None, noise=0.35, mask=None, combine_embeds='average', ref_image=None, patch_file="none"):
        # Clone and convert target image
        tensor_image = image.clone()
        image_np = tensor_to_image(image)
        image_pil = PIL.Image.fromarray(image_np.astype(np.uint8))
        image_pil = image_pil.convert("RGB")
        
        # Load the InfiniteYou model
        infiniteyou_model = self.load_model(adapter_file)
        
        # Set device and dtype
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        device = comfy.model_management.get_torch_device()
        self.dtype = dtype
        self.device = device
        
        # Use provided cn_strength or fall back to weight
        cn_strength = weight if cn_strength is None else cn_strength
        
        # Prepare mask and landmarks
        if mask is None:
            mask_tensor, landmark = self.prepare_mask_and_landmark(infiniteyou_model, image_pil, blur_kernel)
        else:
            # If mask is provided, still get landmarks from the image
            _, landmark = infiniteyou_model.get_face_embed_and_landmark(image_pil)
            mask_tensor = mask
        
        # Prepare conditioning with some default prompts
        prompt = " "
        neg_prompt = "ugly, blurry"
        positive = self.encode_prompt(clip, prompt)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": float(1.5)})
        negative = self.encode_prompt(clip, neg_prompt)
        
        # Move tensor image to device
        tensor_image = tensor_image.to(device, dtype=dtype)
        
        # Prepare inpainting conditioning
        positive, negative, latent = self.prepare_condition_inpainting(positive, negative, tensor_image, vae, mask_tensor)
        
        # Determine whether to use reference image or patch file
        if ref_image is not None:
            # Use reference image
            ref_image_np = tensor_to_image(ref_image)
            ref_image_pil = PIL.Image.fromarray(ref_image_np.astype(np.uint8))
            ref_image_pil = ref_image_pil.convert("RGB")
            
            # Get face embedding from reference image
            face_emb, _ = infiniteyou_model.get_face_embed_and_landmark(ref_image_pil)
            if face_emb is None:
                raise Exception('Reference Image: No face detected.')
            
            # Handle face embedding based on combine_embeds parameter
            clip_embed = face_emb
            if clip_embed.shape[0] > 1:
                if combine_embeds == 'average':
                    clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
                elif combine_embeds == 'norm average':
                    clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)
            
            # Apply noise to embedding if specified
            if noise > 0:
                seed = int(torch.sum(clip_embed).item()) % 1000000007
                torch.manual_seed(seed)
                clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)
            
            # Move model to device and get image embeddings
            infiniteyou_model = infiniteyou_model.to(device, dtype=dtype)
            image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(
                clip_embed.to(device, dtype=dtype), 
                clip_embed_zeroed.to(device, dtype=dtype)
            )
            
            # Create face keypoints visualization
            out = []
            image_kps_pil = draw_kps(image_pil, landmark)
            out.append(image_kps_pil)
            
            # Convert to tensor
            transform = transforms.ToTensor()
            out_tensors = [transform(img) for img in out]
            face_kps_tensor = torch.stack(out_tensors, dim=0).permute(0, 2, 3, 1)
            
        elif patch_file != "none":
            # Handle random patch selection
            if patch_file == "random":
                import random
                patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
                patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
                if not patches:
                    raise Exception("No patch files found in the patch directory.")
                # Pick a random patch
                patch_file = random.choice(patches)
                print(f"Randomly selected patch file: {patch_file}")
            
            # Load the patch data
            patch_path = os.path.join(folder_paths.output_directory, "infiniteyoupatch", patch_file)
            patch_data = torch.load(patch_path)
            
            # Get the embeddings and parameters from the patch data
            image_prompt_embeds = patch_data["image_prompt_embeds"].to(device) if "image_prompt_embeds" in patch_data else None
            uncond_image_prompt_embeds = patch_data["uncond_image_prompt_embeds"].to(device) if "uncond_image_prompt_embeds" in patch_data else None
            face_kps_tensor = patch_data["face_kps"].to(device) if "face_kps" in patch_data else None
            
            # Get control parameters from patch if available
            if "cn_strength" in patch_data:
                cn_strength = patch_data["cn_strength"]
            if "start_at" in patch_data:
                start_at = patch_data["start_at"]
            if "end_at" in patch_data:
                end_at = patch_data["end_at"]
            
            # Make sure we have all the necessary data
            if image_prompt_embeds is None or uncond_image_prompt_embeds is None or face_kps_tensor is None:
                raise Exception("Patch file is missing essential data. Cannot apply face.")
                
            print(f"Applying face from patch file: {patch_file}")
            
        else:
            raise Exception("You must provide either a reference image or select a patch file.")

        
        # Set up the controlnet with the face keypoints
        cnets = {}
        cond_uncond = []
        
        # Process positive and negative conditioning separately
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                # Get the original weight and conditioning dictionary
                weight_cond, cond_dict = t
                
                # Create a copy of the conditioning dictionary to avoid modifying the original
                d = cond_dict.copy()
                
                # Set up the control net
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps_tensor.movedim(-1,1), 
                                                         cn_strength if cn_strength is not None else 1.0, 
                                                         (start_at, end_at), 
                                                         vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                # Apply only the face-specific parts to the conditioning
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                
                # Add the face embeddings
                if is_cond:
                    d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)
                else:
                    d['cross_attn_controlnet'] = uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), 
                                                              dtype=c_net.cond_hint_original.dtype)
                
                # Create the new conditioning entry with the original weight
                n = [weight_cond, d]
                c.append(n)
            
            # Add the processed conditioning to the appropriate list
            cond_uncond.append(c)
            is_cond = False
        
        return (model, cond_uncond[0], cond_uncond[1], latent)

# Add node to mappings
NODE_CLASS_MAPPINGS = {
    "StarInfiniteYouFaceSwapMod": StarInfiniteYouFaceSwapMod
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarInfiniteYouFaceSwapMod": "⭐Star InfiniteYou Face Swap Mod"
}
