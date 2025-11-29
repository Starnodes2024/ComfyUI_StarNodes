import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import PIL.Image
import cv2
from .starinfiniteyou_utils import tensor_to_image, draw_kps
from .starinfiniteyou_core import InfiniteYou

class StarInfiniteYouFaceSwap:
    @classmethod
    def INPUT_TYPES(s):
        patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(patch_dir, exist_ok=True)
        patches = [f for f in os.listdir(patch_dir) if f.endswith((".pt", ".iyou"))]
        patches.sort()
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "weight": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "vae": ("VAE",),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "patch_file": (["none"] + patches,),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "PATCH_DATA")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent", "patch_data")
    FUNCTION = "apply_face_swap"
    CATEGORY = "\u2b50StarNodes/InfiniteYou"

    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")
        model = InfiniteYou(adapter_model_state_dict)
        return model

    def load_patch(self, patch_file):
        patch_path = os.path.join(folder_paths.output_directory, "infiniteyoupatch", patch_file)
        patch_data = torch.load(patch_path)
        return patch_data

    def apply_face_swap(self, adapter_file, control_net, model, clip, image, vae, weight, start_at, end_at, ref_image=None, patch_file="none", mask=None):
        # Handle image shape robustly
        def get_hw_from_image(img):
            # Accepts torch tensor or numpy array, batch or single
            if hasattr(img, 'shape'):
                shape = img.shape
                if len(shape) == 4:  # (B, C, H, W)
                    return shape[2], shape[3]
                elif len(shape) == 3:  # (C, H, W)
                    return shape[1], shape[2]
                elif len(shape) == 2:  # (H, W)
                    return shape[0], shape[1]
            raise Exception("Cannot determine image height/width from input")

        # If ref_image is provided, use it. Otherwise, use patch file.
        if ref_image is not None:
            ref_image_np = tensor_to_image(ref_image)
            ref_image_pil = PIL.Image.fromarray(ref_image_np.astype(np.uint8)).convert("RGB")
            infiniteyou_model = self.load_model(adapter_file)
            dtype = comfy.model_management.unet_dtype()
            if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
            device = comfy.model_management.get_torch_device()
            face_emb, face_kps = infiniteyou_model.get_face_embed_and_landmark(ref_image_pil)
            if face_emb is None:
                raise Exception('Reference Image: No face detected.')
            clip_embed = face_emb
            if clip_embed.shape[0] > 1:
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            noise = 0.35
            if noise > 0:
                seed = int(torch.sum(clip_embed).item()) % 1000000007
                torch.manual_seed(seed)
                clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)
            infiniteyou_model = infiniteyou_model.to(device, dtype=dtype)
            clip_embed = clip_embed.to(device, dtype=dtype)
            clip_embed_zeroed = clip_embed_zeroed.to(device, dtype=dtype)
            image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)
            import torchvision.transforms as T
            out = []
            height, width = get_hw_from_image(image)
            control_image = ref_image_pil.resize((width * 8, height * 8), PIL.Image.BILINEAR)
            image_kps = draw_kps(control_image, face_kps)
            out.append(image_kps)
            transform = T.ToTensor()
            tensor_list = [transform(img) for img in out]
            face_kps_tensor = torch.stack(tensor_list, dim=0).permute([0,2,3,1])
            face_kps_tensor = face_kps_tensor.to(device, dtype=dtype)
            # Generate dummy conditioning lists (same as patch branch)
            dummy_positive = [[1.0, {}]]
            dummy_negative = [[1.0, {}]]
            cond_uncond = [dummy_positive, dummy_negative]
            patch_data = {
                "image_prompt_embeds": image_prompt_embeds.cpu(),
                "uncond_image_prompt_embeds": uncond_image_prompt_embeds.cpu(),
                "face_kps": face_kps_tensor.cpu(),
                "cn_strength": weight,
                "start_at": start_at,
                "end_at": end_at
            }
            return (model, cond_uncond[0], cond_uncond[1], image, patch_data)
        elif patch_file and patch_file != "none":
            patch_data = self.load_patch(patch_file)
            image_prompt_embeds = patch_data.get("image_prompt_embeds")
            uncond_image_prompt_embeds = patch_data.get("uncond_image_prompt_embeds")
            face_kps = patch_data.get("face_kps")
            cn_strength = patch_data.get("cn_strength", weight)
            start_at = patch_data.get("start_at", start_at)
            end_at = patch_data.get("end_at", end_at)
            if image_prompt_embeds is None or uncond_image_prompt_embeds is None or face_kps is None:
                raise Exception("Patch file is missing required face data.")
            cnets = {}
            cond_uncond = []
            is_cond = True
            # For patch_file branch, mimic the original node: create dummy conditioning lists
            dummy_positive = [[1.0, {}]]
            dummy_negative = [[1.0, {}]]
            cond_uncond = [dummy_positive, dummy_negative]
            patch_data_out = {
                "image_prompt_embeds": image_prompt_embeds.cpu(),
                "uncond_image_prompt_embeds": uncond_image_prompt_embeds.cpu(),
                "face_kps": face_kps.cpu(),
                "cn_strength": cn_strength,
                "start_at": start_at,
                "end_at": end_at
            }
            # Encode image to latent using VAE
            latent = vae.encode(image)
            return (model, cond_uncond[0], cond_uncond[1], latent, patch_data_out)
        else:
            raise Exception("You must provide either a reference image or select a patch file.")
