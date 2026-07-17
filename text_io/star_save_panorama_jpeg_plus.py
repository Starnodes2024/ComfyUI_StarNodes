import os
import io
import numpy as np
from PIL import Image
import folder_paths

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[StarSavePanoramaJPEGPlus] Warning: 'opencv-python' not installed. Node will not be available.")

from .star_save_panorama_jpeg import XMP_TEMPLATE


class StarSavePanoramaJPEGPlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "subfolder": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional subfolder inside the base path."}),
                "filename": ("STRING", {"default": "panorama", "multiline": False}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality (1-100)."}),
                "stereo_3d": ("BOOLEAN", {"default": False, "tooltip": "Also save a stereoscopic 3D image (requires depth_map input)."}),
                "stereo_layout": (["SBS", "Top/Bottom"], {"default": "SBS"}),
                "depth_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "Strength of the 3D effect. Higher values shift pixels further."}),
                "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Enable if the depth map uses white for far instead of near."}),
                "gap_fill": (["Inpaint", "None"], {"default": "Inpaint", "tooltip": "Fill gaps created by pixel shifting. Inpaint gives the best quality."}),
            },
            "optional": {
                "base_path": ("STRING", {"forceInput": True, "tooltip": "Base save folder, e.g. the path output of Star Save Image+."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map used to generate the stereoscopic image."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "3d_image",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Saves images as JPEG with panorama XMP metadata. Optionally saves a stereoscopic 3D version (SBS or Top/Bottom) generated from a depth map."

    def _safe_part(self, s):
        invalid = '<>:"|?*\n\r\t'
        return "".join(c for c in (s or "") if c not in invalid).strip().strip("/\\")

    def _make_stereo(self, img_array, depth, depth_scale, invert_depth, gap_fill, stereo_layout):
        height, width = img_array.shape[:2]

        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if invert_depth:
            depth = 1.0 - depth
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

        max_shift_pixels = width * (depth_scale / 500.0)
        pixel_shifts = np.clip((depth * max_shift_pixels).astype(np.int32), 0, width - 1)

        shifted = np.zeros_like(img_array)
        gap_mask = np.ones((height, width), dtype=np.uint8)
        rows = np.arange(height)

        for x in range(width - 1, -1, -1):
            target_x = np.clip(x + pixel_shifts[:, x], 0, width - 1)
            shifted[rows, target_x] = img_array[:, x]
            gap_mask[rows, target_x] = 0

        if gap_fill == "Inpaint":
            shifted = cv2.inpaint(shifted, gap_mask * 255, 3, cv2.INPAINT_TELEA)

        if stereo_layout == "SBS":
            return np.concatenate([img_array, shifted], axis=1)
        return np.concatenate([img_array, shifted], axis=0)

    def save_images(self, images, subfolder="", filename="panorama", quality=95,
                    stereo_3d=False, stereo_layout="SBS", depth_scale=5.0, invert_depth=False,
                    gap_fill="Inpaint", base_path="", depth_map=None, prompt=None, extra_pnginfo=None):
        parts = [self._safe_part(base_path), self._safe_part(subfolder), self._safe_part(filename) or "panorama"]
        filename_prefix = "/".join(p for p in parts if p)

        full_output_folder, filename, counter, out_subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        stereo_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            width, height = img.size

            def inject_pano_xmp(pil_img, qual):
                w, h = pil_img.size
                xmp = XMP_TEMPLATE.format(
                    projection_type="cylindrical",
                    width=w,
                    height=h
                )
                xmp_bytes = xmp.encode("utf-8")
                xmp_tag = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_bytes
                out_bytes = io.BytesIO()
                pil_img.save(out_bytes, format="JPEG", quality=qual)
                jpeg_bytes = out_bytes.getvalue()
                soi = jpeg_bytes[:2]
                rest = jpeg_bytes[2:]
                app1_marker = b'\xFF\xE1' + (len(xmp_tag) + 2).to_bytes(2, 'big') + xmp_tag
                return soi + app1_marker + rest

            new_jpeg = inject_pano_xmp(img, quality)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            with open(os.path.join(full_output_folder, file), "wb") as f:
                f.write(new_jpeg)
            results.append({
                "filename": file,
                "subfolder": out_subfolder,
                "type": self.type
            })

            if stereo_3d and depth_map is not None:
                depth = depth_map[batch_number % depth_map.shape[0]].cpu().numpy()
                stereo = self._make_stereo(img_array, depth, depth_scale, invert_depth, gap_fill, stereo_layout)
                suffix = "SBS" if stereo_layout == "SBS" else "TB"
                stereo_file = f"{filename_with_batch_num}_{counter:05}_{suffix}.jpg"
                stereo_img = Image.fromarray(stereo)
                stereo_jpeg = inject_pano_xmp(stereo_img, quality)
                with open(os.path.join(full_output_folder, stereo_file), "wb") as f:
                    f.write(stereo_jpeg)
                results.append({
                    "filename": stereo_file,
                    "subfolder": out_subfolder,
                    "type": self.type
                })
                stereo_tensor = np.array(stereo_img).astype(np.float32) / 255.0
                stereo_images.append(stereo_tensor)

            counter += 1
        
        import torch
        if stereo_images:
            stereo_output = torch.from_numpy(np.stack(stereo_images))
        else:
            stereo_output = torch.zeros((1, 64, 64, 3))
        
        return {"ui": {"images": results}, "result": (images, stereo_output,)}


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if CV2_AVAILABLE:
    NODE_CLASS_MAPPINGS["StarSavePanoramaJPEGPlus"] = StarSavePanoramaJPEGPlus
    NODE_DISPLAY_NAME_MAPPINGS["StarSavePanoramaJPEGPlus"] = "\u2b50 Star Save Panorama JPG+"
else:
    print("[StarSavePanoramaJPEGPlus] Node not registered due to missing dependency: opencv-python")
