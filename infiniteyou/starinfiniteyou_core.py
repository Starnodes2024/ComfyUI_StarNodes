import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image
import json

from .starinfiniteyou_resampler import Resampler
from .starinfiniteyou_utils import tensor_to_image, resize_and_pad_image, draw_kps, add_noise

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("StarNodes: InsightFace not available. Face features will be disabled.")

try:
    from facexlib.recognition import init_recognition_model
    FACEXLIB_AVAILABLE = True
except ImportError:
    FACEXLIB_AVAILABLE = False
    print("StarNodes: FaceXLib not available. Some face features might be disabled.")

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

# Define model directories
MODELS_DIR = os.path.join(folder_paths.models_dir, "InfiniteYou")
if "InfiniteYou" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["InfiniteYou"]
folder_paths.folder_names_and_paths["InfiniteYou"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

# Ensure output directory exists for patch files
PATCH_DIR = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
os.makedirs(PATCH_DIR, exist_ok=True)

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None, in_settings=None):
    if not INSIGHTFACE_AVAILABLE:
        raise RuntimeError("InsightFace is not available. Cannot extract arcface embedding.")
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0] # [512], normalized
    return face_emb

class InfiniteYou(torch.nn.Module):
    def __init__(self, adapter_model):
        super().__init__()
        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(adapter_model["image_proj"])

        # Load face encoder
        if INSIGHTFACE_AVAILABLE:
            self.app_640 = FaceAnalysis(name='antelopev2', 
                                    root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app_640.prepare(ctx_id=0, det_size=(640, 640))

            self.app_320 = FaceAnalysis(name='antelopev2', 
                                    root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app_320.prepare(ctx_id=0, det_size=(320, 320))

            self.app_160 = FaceAnalysis(name='antelopev2', 
                                    root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app_160.prepare(ctx_id=0, det_size=(160, 160))
        else:
            self.app_640 = None
            self.app_320 = None
            self.app_160 = None

        if FACEXLIB_AVAILABLE:
            self.arcface_model = init_recognition_model('arcface', device='cuda')
        else:
            self.arcface_model = None
        

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=8,
            embedding_dim=512,
            output_dim=4096,
            ff_mult=4
        )
        return image_proj_model
    
    def detect_face(self, id_image_cv2):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not available. Cannot detect face.")
        # Try different detection sizes if face detection fails
        faces = self.app_640.get(id_image_cv2)
        if len(faces) == 0:
            faces = self.app_320.get(id_image_cv2)
        if len(faces) == 0:
            faces = self.app_160.get(id_image_cv2)
        if len(faces) == 0:
            raise ValueError("No face detected in the reference image")
        return faces[0]
    
    def get_face_embed_and_landmark(self, ref_image):
        # Convert PIL Image to CV2 format (BGR)
        ref_image_cv2 = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        
        # Detect face and get landmarks
        face = self.detect_face(ref_image_cv2)
        kps = face.kps
        
        # Get face embedding using arcface
        face_emb = extract_arcface_bgr_embedding(ref_image_cv2, kps, self.arcface_model)
        
        # Reshape the face embedding to match the expected format [1, -1, 512] as in the original implementation
        face_emb = face_emb.clone().unsqueeze(0).float().cuda()
        face_emb = face_emb.reshape([1, -1, 512])
        face_emb = face_emb.to(device='cuda', dtype=torch.bfloat16)
        
        return face_emb, kps
    
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        # Make sure model is on the same device as input tensors
        device = clip_embed.device
        self.image_proj_model = self.image_proj_model.to(device)
        
        # Process the face embedding through the projection model
        with torch.no_grad():
            image_prompt_embeds = self.image_proj_model(clip_embed)
            uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
            
        return image_prompt_embeds, uncond_image_prompt_embeds


# Star InfiniteYou Apply node implementation

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarApplyInfiniteYou": "⭐StarNodes/InfiniteYou/Apply",
    "StarInfiniteYouSaver": "⭐StarNodes/InfiniteYou/Patch Saver",
    "StarInfiniteYouPatch": "⭐StarNodes/InfiniteYou/Patch Loader",
    "StarInfiniteYouFaceSwap": "⭐StarNodes/InfiniteYou/Face Swap",
    "StarInfiniteYouFaceCombine": "⭐StarNodes/InfiniteYou/Face Combine"
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
            },
            # No optional parameters to match original implementation
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

    def apply_infinite_you(self, adapter_file, control_net, ref_image, model, positive, negative, start_at, end_at, vae, latent_image, fixed_face_pose, weight=0.99, balance=0.5, noise=0.35, cn_strength=None):
        ref_image = tensor_to_image(ref_image)
        ref_image = PIL.Image.fromarray(ref_image.astype(np.uint8))
        ref_image = ref_image.convert("RGB")

        # Load resampler model
        infiniteyou_model = self.load_model(adapter_file)

        # Extract face embedding and landmarks
        face_emb, face_kps = infiniteyou_model.get_face_embed_and_landmark(ref_image)

        # Add noise to embedding if specified
        if noise > 0:
            face_emb = add_noise(face_emb, noise)

        # Create device tensors
        device = comfy.model_management.get_torch_device()
        clip_embed = face_emb.unsqueeze(0).to(device=device)
        clip_embed_zeroed = torch.zeros_like(clip_embed)

        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)

        # Create face keypoints visualization for controlnet
        face_kps_image = draw_kps(ref_image, face_kps)
        face_kps_tensor = torch.from_numpy(np.array(face_kps_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

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
                                                            cn_strength if cn_strength is not None else 1.0, 
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
            "cn_strength": cn_strength if cn_strength is not None else 1.0,
            "start_at": start_at,
            "end_at": end_at
        }

        return (model, cond_uncond[0], cond_uncond[1], latent_image, patch_data)


# Star InfiniteYou Patch Saver node implementation
class StarInfiniteYouSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "patch_data": ("PATCH_DATA",),
                "save_name": ("STRING", {"default": "my_patch"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_patch"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def save_patch(self, patch_data, save_name):
        import os
        import torch
        import json
        import copy
        
        # Ensure the filename has no invalid characters
        save_name = "".join(c for c in save_name if c.isalnum() or c in "_-").strip()
        if not save_name:
            save_name = "unnamed_patch"
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full path for the patch file
        patch_path = os.path.join(output_dir, f"{save_name}.pt")
        
        # Create a deep copy of the patch data to avoid modifying the original
        processed_data = copy.deepcopy(patch_data)
        
        # Process the conditioning data to make it serializable
        if "processed_positive" in processed_data and "processed_negative" in processed_data:
            # Create serializable versions of the conditioning
            processed_data["processed_positive"] = self.make_serializable(processed_data["processed_positive"])
            processed_data["processed_negative"] = self.make_serializable(processed_data["processed_negative"])
        
        # Convert any tensors to CPU before saving
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                processed_data[key] = value.detach().cpu()
        
        # Save the patch data
        try:
            torch.save(processed_data, patch_path)
            print(f"InfiniteYou patch saved to: {patch_path}")
        except Exception as e:
            print(f"Error saving patch: {str(e)}")
            # Fallback to saving just the embeddings if full conditioning can't be saved
            try:
                fallback_data = {
                    "image_prompt_embeds": processed_data["image_prompt_embeds"].detach().cpu() if isinstance(processed_data["image_prompt_embeds"], torch.Tensor) else processed_data["image_prompt_embeds"],
                    "uncond_image_prompt_embeds": processed_data["uncond_image_prompt_embeds"].detach().cpu() if isinstance(processed_data["uncond_image_prompt_embeds"], torch.Tensor) else processed_data["uncond_image_prompt_embeds"],
                    "face_kps": processed_data["face_kps"].detach().cpu() if isinstance(processed_data["face_kps"], torch.Tensor) else processed_data["face_kps"],
                    "cn_strength": processed_data["cn_strength"],
                    "start_at": processed_data["start_at"],
                    "end_at": processed_data["end_at"]
                }
                torch.save(fallback_data, patch_path)
                print(f"Saved fallback patch data (without processed conditioning) to: {patch_path}")
            except Exception as e2:
                print(f"Failed to save even fallback data: {str(e2)}")
        
        return {}
    
    def make_serializable(self, conditioning):
        """Convert conditioning to a serializable format by removing non-serializable objects"""
        serializable_conditioning = []
        
        for cond in conditioning:
            # Each conditioning item is a tuple (weight, dict)
            weight, cond_dict = cond
            
            # Create a new dict with only serializable items
            serializable_dict = {}
            
            # Copy serializable items
            for k, v in cond_dict.items():
                # Skip known non-serializable keys
                if k in ['control']:
                    continue
                    
                # Try to keep tensors and basic types
                if isinstance(v, (int, float, str, bool, torch.Tensor)):
                    serializable_dict[k] = v
                elif v is None:
                    serializable_dict[k] = None
                # Skip complex objects
            
            # Add to the new conditioning list
            serializable_conditioning.append((weight, serializable_dict))
        
        return serializable_conditioning


# Star InfiniteYou Patch Loader node implementation
class StarInfiniteYouPatch:
    @classmethod
    def INPUT_TYPES(s):
        # Get list of patch files
        patch_dir = os.path.join(folder_paths.output_directory, "infiniteyoupatch")
        os.makedirs(patch_dir, exist_ok=True)
        patches = [f for f in os.listdir(patch_dir) if f.endswith(".pt")]
        # Sort patches alphabetically for better organization
        patches.sort()
        
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "patch_file": (["none"] + patches,),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_patch"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def apply_patch(self, control_net, model, positive, negative, vae, latent_image, patch_file):
        if patch_file == "none":
            # If no patch file is selected, just return the inputs unchanged
            return (model, positive, negative, latent_image)
        
        # Load the patch data
        patch_path = os.path.join(folder_paths.output_directory, "infiniteyoupatch", patch_file)
        patch_data = torch.load(patch_path)
        
        # Get the embeddings and parameters from the patch data
        device = comfy.model_management.get_torch_device()
        image_prompt_embeds = patch_data["image_prompt_embeds"].to(device) if "image_prompt_embeds" in patch_data else None
        uncond_image_prompt_embeds = patch_data["uncond_image_prompt_embeds"].to(device) if "uncond_image_prompt_embeds" in patch_data else None
        face_kps = patch_data["face_kps"] if "face_kps" in patch_data else None
        
        # Get control parameters
        cn_strength = patch_data.get("cn_strength", 1.0)
        start_at = patch_data.get("start_at", 0.0)
        end_at = patch_data.get("end_at", 1.0)
        
        # Make sure we have all the necessary data
        if image_prompt_embeds is None or uncond_image_prompt_embeds is None or face_kps is None:
            print("Warning: Patch file is missing essential data. Cannot apply face.")
            return (model, positive, negative, latent_image)
        
        print(f"Applying face from patch file: {patch_file} to user conditioning")
        
        # Set up the controlnet with the face keypoints
        cnets = {}
        cond_uncond = []
        
        # Process positive and negative conditioning separately
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                # Get the original weight and conditioning dictionary
                weight, cond_dict = t
                
                # Create a copy of the conditioning dictionary to avoid modifying the original
                d = cond_dict.copy()
                
                # Set up the control net
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
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
                n = [weight, d]
                c.append(n)
            
            # Add the processed conditioning to the appropriate list
            cond_uncond.append(c)
            is_cond = False
        
        return(model, cond_uncond[0], cond_uncond[1], latent_image)


# Star InfiniteYou Face Swap node implementation
class StarInfiniteYouFaceSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "ref_image": ("IMAGE",),
                "image": ("IMAGE",),
                "clip": ("CLIP",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "vae": ("VAE",),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "blur_kernel": ("INT", {"default": 9, "min": 1, "max": 31, "step": 2}),
            },
            "optional": {
                "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_face_swap"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model
    
    def encode_prompt(self, clip, text):
        # Encode text prompt using CLIP
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    def prepare_condition_inpainting(self, positive, negative, pixels, vae, mask):
        # Prepare conditioning for inpainting
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        pixels = pixels[:, :x, :y, :]
        mask = mask[:, :x, :y, :]
        
        # Encode the image with VAE
        pixel_samples = pixels.movedim(3, 1)
        pixel_samples = pixel_samples * 2.0 - 1.0
        
        # Encode masked image
        vae_encoder = vae.first_stage_model.encode
        mask_pixel_samples = pixel_samples * (1.0 - mask.movedim(3, 1))
        
        # Get latent representation
        input_dtype = pixel_samples.dtype
        with torch.no_grad():
            with torch.autocast("cuda"):
                latent = vae_encoder(mask_pixel_samples.to(vae.dtype))
                if isinstance(latent, torch.distributions.Distribution):
                    latent = latent.mode() 
        
        latent = latent.to(input_dtype)
        
        # Prepare latent mask
        latent_mask = torch.nn.functional.interpolate(mask.movedim(3, 1), size=(latent.shape[2], latent.shape[3]))
        
        # Add inpainting conditioning to positive and negative
        new_positive = []
        new_negative = []
        
        for i, (cond, cond_dict) in enumerate(positive):
            d = cond_dict.copy()
            d["latent_mask"] = latent_mask
            d["latent"] = latent
            d["mask"] = mask.movedim(3, 1)
            new_positive.append([cond, d])
        
        for i, (cond, cond_dict) in enumerate(negative):
            d = cond_dict.copy()
            d["latent_mask"] = latent_mask
            d["latent"] = latent
            d["mask"] = mask.movedim(3, 1)
            new_negative.append([cond, d])
        
        return new_positive, new_negative
    
    def prepare_mask_and_landmark(self, image, blur_kernel):
        # Convert tensor to PIL image for face detection
        image_pil = PIL.Image.fromarray(tensor_to_image(image))
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Load face detection model
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not available. Cannot detect face.")
        
        import folder_paths
        
        INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
        app = FaceAnalysis(name='antelopev2', root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Detect faces
        faces = app.get(image_cv2)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Get the first face
        face = faces[0]
        kps = face.kps
        
        # Create mask from face landmarks
        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
        hull = cv2.convexHull(np.array(kps).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Apply Gaussian blur to the mask edges
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        mask = mask / 255.0
        
        # Convert mask to tensor format
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).float()
        
        return mask_tensor, kps

    def apply_face_swap(self, control_net, model, ref_image, image, clip, start_at, end_at, vae, adapter_file, blur_kernel=9, noise=0.35, cn_strength=None):
        # Load the adapter model
        infiniteyou_model = self.load_model(adapter_file)
        
        # Prepare reference image
        ref_image = tensor_to_image(ref_image)
        ref_image = PIL.Image.fromarray(ref_image.astype(np.uint8))
        ref_image = ref_image.convert("RGB")

        # Extract face embedding and landmarks
        face_emb, face_kps = infiniteyou_model.get_face_embed_and_landmark(ref_image)
        
        # Add noise to embedding if specified
        if noise > 0:
            face_emb = add_noise(face_emb, noise)
        
        # Prepare mask
        mask, _ = self.prepare_mask_and_landmark(image, blur_kernel)
        
        # Create device tensors
        device = comfy.model_management.get_torch_device()
        clip_embed = face_emb.unsqueeze(0).to(device=device)
        clip_embed_zeroed = torch.zeros_like(clip_embed)
        
        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)
        
        # Prepare inpainting conditioning
        positive = self.encode_prompt(clip, "face")
        negative = self.encode_prompt(clip, "")
        
        # Prepare inpainting conditioning
        positive, negative = self.prepare_condition_inpainting(positive, negative, image, vae, mask)
        
        # Set up controlnet with face keypoints
        cnets = {}
        cond_uncond = []
        
        # Create face keypoints tensor for controlnet
        face_kps_tensor = torch.from_numpy(np.array(face_kps)).unsqueeze(0).unsqueeze(-1).float()
        
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
                                                          cn_strength if cn_strength is not None else 1.0, 
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
        
        # Create latent from image
        latent = {"samples": vae.encode(image.permute(0, 3, 1, 2) * 2 - 1)}
        
        return (model, cond_uncond[0], cond_uncond[1], latent)


# Star InfiniteYou Face Combine node implementation
class StarInfiniteYouFaceCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "latent_image": ("LATENT",),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"),),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "vae": ("VAE",),
                "fixed_face_pose": ("BOOLEAN", {"default": False, "tooltip": "Fix the face pose from reference image."}),
            },
            "optional": {
                "noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_face_combine"
    CATEGORY = "⭐StarNodes/InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model

    def apply_face_combine(self, control_net, model, positive, negative, ref_image_1, ref_image_2, latent_image, adapter_file, balance, start_at, end_at, vae, fixed_face_pose, noise=0.35, cn_strength=None):
        # Load the adapter model
        infiniteyou_model = self.load_model(adapter_file)
        
        # Prepare reference images
        ref_image_1 = tensor_to_image(ref_image_1)
        ref_image_1 = PIL.Image.fromarray(ref_image_1.astype(np.uint8))
        ref_image_1 = ref_image_1.convert("RGB")
        
        ref_image_2 = tensor_to_image(ref_image_2)
        ref_image_2 = PIL.Image.fromarray(ref_image_2.astype(np.uint8))
        ref_image_2 = ref_image_2.convert("RGB")
        
        # Extract face embeddings and landmarks
        face_emb_1, face_kps_1 = infiniteyou_model.get_face_embed_and_landmark(ref_image_1)
        face_emb_2, face_kps_2 = infiniteyou_model.get_face_embed_and_landmark(ref_image_2)
        
        # Add noise to embeddings if specified
        if noise > 0:
            face_emb_1 = add_noise(face_emb_1, noise)
            face_emb_2 = add_noise(face_emb_2, noise)
        
        # Combine face embeddings based on balance parameter
        face_emb = face_emb_1 * balance + face_emb_2 * (1 - balance)
        
        # Combine face keypoints based on balance parameter
        face_kps = face_kps_1 * balance + face_kps_2 * (1 - balance)
        
        # Create device tensors
        device = comfy.model_management.get_torch_device()
        clip_embed = face_emb.unsqueeze(0).to(device=device)
        clip_embed_zeroed = torch.zeros_like(clip_embed)
        
        # Get image embeddings
        image_prompt_embeds, uncond_image_prompt_embeds = infiniteyou_model.get_image_embeds(clip_embed, clip_embed_zeroed)
        
        # Create face keypoints visualization for controlnet
        combined_face_pil = PIL.Image.blend(ref_image_1, ref_image_2, 1 - balance)
        face_kps_image = draw_kps(combined_face_pil, face_kps)
        face_kps_tensor = torch.from_numpy(np.array(face_kps_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
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
                                                          cn_strength if cn_strength is not None else 1.0, 
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
        
        return (model, cond_uncond[0], cond_uncond[1], latent_image)
