"""
Star Face Detailer+
-------------------
A user-friendly, self-contained face detailing node for ComfyUI.

Inspired by FaceDetailer from ComfyUI-Impact-Pack (credit: ltdrdata),
but reworked to be simpler and more powerful:

  * No external detector loader nodes - bbox/segm models are picked from
    dropdowns (models/ultralytics/bbox and models/ultralytics/segm).
  * Prompt inputs are widgets - only image / model / clip / vae are inputs.
  * Single output: the finished, refined image.
  * Max Faces limit for big group photos.
  * Per-face LoRA slots (up to 5) - face #1 can use LoRA 1, face #2 LoRA 2, ...
  * No SAM.
  * Live DOM progress bar + preview of the face currently being detailed.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.samplers
import comfy.utils
import model_management
import nodes
from server import PromptServer

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[StarFaceDetailerPlus] Warning: 'ultralytics' not installed. Node will not be available.")

PROGRESS_EVENT = "starface.detailer.progress"

# Cache loaded YOLO models so repeat runs are fast.
_YOLO_CACHE = {}


# --------------------------------------------------------------------------- #
#  Model / path helpers
# --------------------------------------------------------------------------- #

def _ultra_key(sub):
    """Return (folder_paths key, prefix) that matches models/ultralytics/<sub>."""
    if f"ultralytics_{sub}" in folder_paths.folder_names_and_paths:
        return f"ultralytics_{sub}", ""
    if "ultralytics" in folder_paths.folder_names_and_paths:
        return "ultralytics", f"{sub}/"
    return None, None


def _ultra_files(sub):
    key, prefix = _ultra_key(sub)
    if key is None:
        return []
    try:
        files = folder_paths.get_filename_list(key)
    except Exception:
        return []
    if prefix:
        files = [f[len(prefix):] for f in files if f.startswith(prefix)]
    return sorted(files)


def _ultra_path(sub, name):
    key, prefix = _ultra_key(sub)
    if key is None:
        return None
    return folder_paths.get_full_path(key, prefix + name)


def _get_yolo(path):
    if path in _YOLO_CACHE:
        return _YOLO_CACHE[path]
    model = YOLO(path)
    _YOLO_CACHE[path] = model
    return model


# --------------------------------------------------------------------------- #
#  Small utilities
# --------------------------------------------------------------------------- #

def _tensor_to_pil(t):
    arr = (t[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _upscale(t, width, height, method="lanczos"):
    """t: [B,H,W,C] -> resized [B,H,W,C]"""
    s = t.movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, method, "disabled")
    return s.movedim(1, -1)


def _feather_mask(mask_np, feather):
    """Gaussian-blur a float32 HxW mask. Falls back to box blur without cv2."""
    if feather <= 0:
        return mask_np
    k = max(3, int(feather) * 2 + 1)
    if k % 2 == 0:
        k += 1
    try:
        import cv2
        return cv2.GaussianBlur(mask_np, (k, k), 0).clip(0.0, 1.0)
    except ImportError:
        t = torch.from_numpy(mask_np)[None, None]
        # simple separable box blur approximation
        for _ in range(2):
            t = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(t, (k // 2,) * 4, mode="reflect"),
                kernel_size=k, stride=1,
            )
        return t[0, 0].numpy().clip(0.0, 1.0)


def _preview_data_url(t):
    """[1,H,W,C] tensor -> downscaled PNG data URL for the DOM preview."""
    pil = _tensor_to_pil(t)
    pil.thumbnail((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _send_progress(unique_id, **payload):
    if not unique_id:
        return
    try:
        PromptServer.instance.send_sync(
            PROGRESS_EVENT, {"node": str(unique_id), **payload}
        )
    except Exception:
        pass  # never let UI messaging break execution


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #

def _detect_bboxes(pil_img, model_path, threshold):
    model = _get_yolo(model_path)
    device = model_management.get_torch_device()
    results = model.predict(
        source=np.asarray(pil_img), conf=threshold, device=device, verbose=False
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return r.boxes.xyxy.cpu().numpy()


def _segm_mask(pil_crop, model_path, threshold, face_box=None):
    """Segmentation mask of the TARGET face inside the crop, float32 HxW.

    face_box: (x1, y1, x2, y2) of the face being processed, in crop pixels.
    The crop usually contains several people (crop_factor context), so we must
    NOT simply take the largest segment. We pick the detected segment whose box
    best matches the face box (highest IoU), falling back to a segment that
    contains the face-box center. Returns None when nothing matches - the
    caller then falls back to the plain bbox rect mask.
    """
    model = _get_yolo(model_path)
    device = model_management.get_torch_device()
    results = model.predict(
        source=np.asarray(pil_crop), conf=threshold, device=device, verbose=False
    )
    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None

    masks = r.masks.data.cpu().numpy()
    idx = None

    if face_box is not None and r.boxes is not None and len(r.boxes) == len(masks):
        fx1, fy1, fx2, fy2 = face_box
        farea = max((fx2 - fx1) * (fy2 - fy1), 1.0)
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        best_iou, best_idx, center_idx = 0.0, None, None
        for j, (bx1, by1, bx2, by2) in enumerate(r.boxes.xyxy.cpu().numpy()):
            ix1, iy1 = max(fx1, bx1), max(fy1, by1)
            ix2, iy2 = min(fx2, bx2), min(fy2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = farea + max((bx2 - bx1) * (by2 - by1), 1.0) - inter
            iou = inter / max(union, 1.0)
            if iou > best_iou:
                best_iou, best_idx = iou, j
            if bx1 <= fcx <= bx2 and by1 <= fcy <= by2:
                center_idx = j
        idx = best_idx if (best_idx is not None and best_iou > 0.01) else center_idx

    if idx is None:
        return None

    m = masks[idx]
    m = np.asarray(
        Image.fromarray((m * 255).astype(np.uint8)).resize(pil_crop.size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return (m > 0.5).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Node
# --------------------------------------------------------------------------- #

class StarFaceDetailerPlus:
    @classmethod
    def INPUT_TYPES(cls):
        bbox_models = _ultra_files("bbox") or ["(no models found in models/ultralytics/bbox)"]
        segm_models = ["none"] + (_ultra_files("segm") or [])
        loras = ["none"] + sorted(folder_paths.get_filename_list("loras"))

        required = {
            "image": ("IMAGE",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),

            # --- detection ---
            "bbox_model": (bbox_models, {
                "default": "face_yolov8n_v2.pt",
                "tooltip": "Ultralytics BBOX detector used to find faces. "
                           "Loaded from models/ultralytics/bbox."}),
            "segm_model": (segm_models, {
                "default": "face_yolov8n-seg2_60.pt",
                "tooltip": "Optional Ultralytics SEGM detector for a precise face mask. "
                           "Loaded from models/ultralytics/segm. 'none' = use the plain bbox."}),
            "max_faces": ("INT", {"default": 5, "min": 1, "max": 64, "step": 1,
                "tooltip": "Maximum number of faces that will be detailed. "
                           "Handy for group photos - extra faces are left untouched."}),
            "face_order": (["largest first", "left to right", "right to left", "top to bottom"], {
                "default": "largest first",
                "tooltip": "Which faces get priority when Max Faces limits the count."}),

            # --- prompts ---
            "positive": ("STRING", {"multiline": True, "default": "",
                "tooltip": "Positive prompt used for every face refinement."}),
            "negative": ("STRING", {"multiline": True, "default": "",
                "tooltip": "Negative prompt used for every face refinement."}),

            # --- sizing ---
            "guide_size": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8,
                "tooltip": "Target resolution before detailing (see guide_size_for). "
                           "Raise it for tiny faces in group shots."}),
            "guide_size_for": (["face (bbox)", "crop region"], {
                "default": "face (bbox)",
                "tooltip": "face (bbox): the FACE itself is scaled to guide_size - "
                           "best for small faces. crop region: the whole crop "
                           "(face + context) is scaled to guide_size."}),
            "max_size": ("INT", {"default": 1328, "min": 64, "max": 4096, "step": 8,
                "tooltip": "Hard cap on the working resolution (VRAM/speed safety)."}),
            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                "tooltip": "How much context around the face is included. "
                           "Crop size = face size x crop_factor."}),
            "bbox_threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Minimum detector confidence for a face to be accepted."}),
            "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1,
                "tooltip": "Grow (or shrink) the detected face box by this many pixels per side."}),
            "drop_size": ("INT", {"default": 10, "min": 1, "max": 512, "step": 1,
                "tooltip": "Faces smaller than this (pixels) are ignored."}),

            # --- sampling ---
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                "tooltip": "Noise seed. The same seed is reused for every face in one run."}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1,
                "tooltip": "Sampler steps per face."}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                "tooltip": "Classifier-free guidance scale."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                "default": "euler",
                "tooltip": "Sampler used for the face refinement."}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                "default": "simple",
                "tooltip": "Scheduler used for the face refinement."}),
            "denoise": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Inpaint strength. Lower = closer to the original face, "
                           "higher = more regeneration."}),
            "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1,
                "tooltip": "Softness of the mask edge when blending the refined face back."}),
        }

        # --- per-face LoRA slots ---
        for i in range(1, 6):
            required[f"lora_{i}"] = (loras, {
                "tooltip": f"LoRA applied only while detailing face #{i}. 'none' = disabled."})
            required[f"lora_strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                "tooltip": f"Strength of LoRA {i} (applied to model and CLIP)."})

        return {
            "required": required,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"
    CATEGORY = "⭐StarNodes/Sampler"
    DESCRIPTION = ("Detects faces and re-renders each one at higher quality. "
                   "Built-in detectors, per-face LoRAs, max-face limit and live preview.")

    # ------------------------------------------------------------------ #

    def detail(self, image, model, clip, vae,
               bbox_model, segm_model, max_faces, face_order,
               positive, negative,
               guide_size, guide_size_for, max_size, crop_factor,
               bbox_threshold, bbox_dilation, drop_size,
               seed, steps, cfg, sampler_name, scheduler, denoise, feather,
               lora_1, lora_strength_1, lora_2, lora_strength_2,
               lora_3, lora_strength_3, lora_4, lora_strength_4,
               lora_5, lora_strength_5,
               unique_id=None):

        bbox_path = _ultra_path("bbox", bbox_model)
        if not bbox_path:
            raise FileNotFoundError(
                f"BBOX model '{bbox_model}' not found. Put Ultralytics bbox models "
                "into models/ultralytics/bbox and refresh.")
        segm_path = None
        if segm_model != "none":
            segm_path = _ultra_path("segm", segm_model)
            if not segm_path:
                raise FileNotFoundError(
                    f"SEGM model '{segm_model}' not found in models/ultralytics/segm.")

        lora_slots = [
            (lora_1, lora_strength_1), (lora_2, lora_strength_2),
            (lora_3, lora_strength_3), (lora_4, lora_strength_4),
            (lora_5, lora_strength_5),
        ]

        out = []
        for b in range(image.shape[0]):
            out.append(self._detail_one(
                image[b:b + 1], model, clip, vae,
                bbox_path, segm_path, max_faces, face_order,
                positive, negative,
                guide_size, guide_size_for, max_size, crop_factor,
                bbox_threshold, bbox_dilation, drop_size,
                seed, steps, cfg, sampler_name, scheduler, denoise, feather,
                lora_slots, unique_id))

        return (torch.cat(out, dim=0),)

    # ------------------------------------------------------------------ #

    def _detail_one(self, img, model, clip, vae,
                    bbox_path, segm_path, max_faces, face_order,
                    positive, negative,
                    guide_size, guide_size_for, max_size, crop_factor,
                    bbox_threshold, bbox_dilation, drop_size,
                    seed, steps, cfg, sampler_name, scheduler, denoise, feather,
                    lora_slots, unique_id):

        _, H, W, _ = img.shape
        pil = _tensor_to_pil(img)

        # ---- detect & filter faces ------------------------------------ #
        boxes = _detect_bboxes(pil, bbox_path, bbox_threshold)
        faces = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0.0, x1 - bbox_dilation); y1 = max(0.0, y1 - bbox_dilation)
            x2 = min(float(W), x2 + bbox_dilation); y2 = min(float(H), y2 + bbox_dilation)
            if max(x2 - x1, y2 - y1) >= drop_size:
                faces.append((x1, y1, x2, y2))

        if face_order == "largest first":
            faces.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        elif face_order == "left to right":
            faces.sort(key=lambda b: (b[0] + b[2]) / 2)
        elif face_order == "right to left":
            faces.sort(key=lambda b: (b[0] + b[2]) / 2, reverse=True)
        else:  # top to bottom
            faces.sort(key=lambda b: (b[1] + b[3]) / 2)
        faces = faces[:max_faces]

        if not faces:
            _send_progress(unique_id, status="done", face=0, total=0)
            return img

        total = len(faces)
        pbar = comfy.utils.ProgressBar(total)
        base = img.clone()

        # ---- refine each face ----------------------------------------- #
        for i, (x1, y1, x2, y2) in enumerate(faces):
            # square crop with context around the face
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = max(x2 - x1, y2 - y1) * crop_factor
            cx1 = int(round(min(max(cx - side / 2.0, 0), max(W - side, 0))))
            cy1 = int(round(min(max(cy - side / 2.0, 0), max(H - side, 0))))
            cx2 = int(round(min(cx1 + side, W))); cy2 = int(round(min(cy1 + side, H)))
            cw, ch = cx2 - cx1, cy2 - cy1

            crop = img[:, cy1:cy2, cx1:cx2, :]

            # ---- working resolution: ratio-preserving upscale ---------------- #
            # Scaling is uniform (same factor for W and H), so the aspect ratio
            # is preserved by definition. In "face (bbox)" mode the FACE itself
            # is scaled up to guide_size pixels - this is what saves small faces
            # in group shots. Sizes snap to multiples of 8 for the VAE latent.
            crop_side = max(cw, ch)
            if guide_size_for == "face (bbox)":
                face_side = max(x2 - x1, y2 - y1)
                scale = guide_size / max(face_side, 1.0)
            else:  # crop region
                scale = guide_size / crop_side if crop_side < guide_size else 1.0
            scale = min(scale, max_size / crop_side)
            tw = max(64, (int(round(cw * scale)) // 8) * 8)
            th = max(64, (int(round(ch * scale)) // 8) * 8)
            work = _upscale(crop, tw, th) if (tw != cw or th != ch) else crop

            # mask: segm mask of THIS face inside the crop, or the bbox rect.
            # The crop contains context (often other people), so the segm mask
            # must be matched to this face's box - never just the biggest blob.
            sx, sy = tw / cw, th / ch
            mask_np = None
            if segm_path:
                face_box = ((x1 - cx1) * sx, (y1 - cy1) * sy,
                            (x2 - cx1) * sx, (y2 - cy1) * sy)
                mask_np = _segm_mask(_tensor_to_pil(work), segm_path,
                                     bbox_threshold, face_box)
            if mask_np is None:
                mask_np = np.zeros((th, tw), dtype=np.float32)
                mx1 = int((x1 - cx1) * sx); my1 = int((y1 - cy1) * sy)
                mx2 = int(np.ceil((x2 - cx1) * sx)); my2 = int(np.ceil((y2 - cy1) * sy))
                mask_np[max(my1, 0):min(my2, th), max(mx1, 0):min(mx2, tw)] = 1.0

            # Feather is applied at WORKING resolution but scaled by the upscale
            # factor, so after the mask is downscaled back to the original crop
            # size the blend edge is exactly `feather` pixels wide in the final
            # image - no hard seams on heavily upscaled small faces.
            feather_work = feather * (tw / max(cw, 1) + th / max(ch, 1)) / 2.0
            mask_np = _feather_mask(mask_np, feather_work)
            mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W]

            # live preview: the tight face box (with a small margin), not the
            # whole inpaint crop, so you always see the face being processed
            pad = 0.18 * max(x2 - x1, y2 - y1)
            px1 = int(max(x1 - pad, 0)); py1 = int(max(y1 - pad, 0))
            px2 = int(min(x2 + pad, W)); py2 = int(min(y2 + pad, H))
            face_preview = img[:, py1:py2, px1:px2, :]
            _send_progress(unique_id, status="processing", face=i + 1, total=total,
                           preview=_preview_data_url(face_preview))

            # per-face LoRA (base model/clip stay untouched)
            m, c = model, clip
            lora_log = "none"
            if i < len(lora_slots):
                lora_name, lora_strength = lora_slots[i]
                if lora_name != "none" and lora_strength != 0.0:
                    m, c = nodes.NODE_CLASS_MAPPINGS["LoraLoader"]().load_lora(
                        model, clip, lora_name, lora_strength, lora_strength)
                    lora_log = f"{lora_name} @ {lora_strength:.2f}"

            print(f"[StarFaceDetailerPlus] face {i + 1}/{total}: "
                  f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), "
                  f"crop={cw}x{ch} -> work={tw}x{th}, "
                  f"mask cover={float(mask_np.mean()) * 100:.1f}%, lora={lora_log}")

            pos = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]().encode(c, positive)[0]
            neg = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]().encode(c, negative)[0]

            # Impact-style latent: encode the untouched pixels and steer the
            # sampler with a noise_mask at latent resolution. (VAEEncodeForInpaint
            # grey-fills the masked area first, which leaks into the result when
            # denoise < 1.0.)
            latent = dict(nodes.NODE_CLASS_MAPPINGS["VAEEncode"]().encode(vae, work)[0])
            mh, mw = mask.shape[-2] // 8, mask.shape[-1] // 8
            noise_mask = torch.nn.functional.interpolate(
                mask.reshape(1, 1, mask.shape[-2], mask.shape[-1]),
                size=(mh, mw), mode="area",
            ).reshape(1, mh, mw)
            latent["noise_mask"] = noise_mask.to(latent["samples"].device)
            sampled = nodes.NODE_CLASS_MAPPINGS["KSampler"]().sample(
                m, seed, steps, cfg, sampler_name, scheduler,
                pos, neg, latent, denoise)[0]
            refined = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]().decode(vae, sampled)[0]

            # blend back with the feathered mask
            refined = _upscale(refined, cw, ch)
            m_small = torch.from_numpy(
                np.asarray(Image.fromarray((mask_np * 255).astype(np.uint8))
                           .resize((cw, ch), Image.BILINEAR), dtype=np.float32) / 255.0
            ).to(refined.device).unsqueeze(-1)  # [H,W,1]
            base[:, cy1:cy2, cx1:cx2, :] = refined * m_small + base[:, cy1:cy2, cx1:cx2, :] * (1.0 - m_small)

            pbar.update_absolute(i + 1)

        _send_progress(unique_id, status="done", face=total, total=total)
        return base


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if ULTRALYTICS_AVAILABLE:
    NODE_CLASS_MAPPINGS["StarFaceDetailerPlus"] = StarFaceDetailerPlus
    NODE_DISPLAY_NAME_MAPPINGS["StarFaceDetailerPlus"] = "⭐ Star Face Detailer+"
