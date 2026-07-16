import folder_paths
import os
import math
import numpy as np
from PIL import Image
import torch


class StarPanoramaViewerPro:
    RATIO_MAP = {
        "custom": None,
        "1:1": (1, 1),
        "1:2": (1, 2),
        "3:4": (3, 4),
        "2:3": (2, 3),
        "5:7": (5, 7),
        "9:16": (9, 16),
        "9:21": (9, 21),
        "10:16": (10, 16),
        "4:3": (4, 3),
        "16:10": (16, 10),
        "3:2": (3, 2),
        "2:1": (2, 1),
        "7:5": (7, 5),
        "16:9": (16, 9),
        "21:9": (21, 9),
    }

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        ratio_labels = list(s.RATIO_MAP.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["Mono", "SBS", "Top/Bottom"], {"default": "Top/Bottom"}),
                "create_video_frames": ("BOOLEAN", {"default": False, "label": "Create Video Frames"}),
                "resolution": (["HD (1280x720)", "Full HD (1920x1080)"], {"default": "Full HD (1920x1080)"}),
                "ratio": (ratio_labels, {"default": "16:9"}),
                "custom_ratio": ("STRING", {"default": "21:9", "multiline": False}),
                "framerate": ([24, 25, 30, 50, 60], {"default": 30}),
                "direction": (["Right to Left", "Left to Right"], {"default": "Right to Left"}),
                "num_loops": ("INT", {"default": 1, "min": 1, "max": 100}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "speed": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 360.0, "step": 1.0}),
            },
            "optional": {
                "depth_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "view_panorama_pro"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "360-degree panorama viewer with parallax effect that exports an image batch for video creation."

    @staticmethod
    def _parse_custom_ratio(text):
        try:
            txt = (text or "").strip()
            if ":" in txt:
                parts = txt.split(":", 1)
            elif "x" in txt.lower():
                parts = txt.lower().split("x", 1)
            else:
                return None
            if len(parts) == 2:
                w_val = int(parts[0].strip())
                h_val = int(parts[1].strip())
                if w_val > 0 and h_val > 0:
                    return (w_val, h_val)
        except Exception:
            pass
        return None

    def _get_export_dims(self, resolution, ratio, custom_ratio):
        base_width = 1920 if "Full HD" in resolution else 1280

        if ratio == "custom":
            parsed = self._parse_custom_ratio(custom_ratio)
            if parsed:
                rw, rh = parsed
            else:
                rw, rh = 16, 9
        else:
            r = self.RATIO_MAP.get(ratio)
            if r:
                rw, rh = r
            else:
                rw, rh = 16, 9

        width = base_width
        height = int(round(width * rh / rw))
        width = width - (width % 2)
        height = height - (height % 2)
        return width, height

    @staticmethod
    def _equirect_to_perspective(equi_np, yaw_deg, pitch_deg, fov_deg, out_w, out_h):
        equi_h, equi_w = equi_np.shape[:2]

        fov_h_rad = math.radians(fov_deg)
        half_w = math.tan(fov_h_rad / 2.0)
        half_h = half_w * (float(out_h) / float(out_w))

        u = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * 2.0 - 1.0
        v = 1.0 - (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * 2.0
        U, V = np.meshgrid(u * half_w, v * half_h)

        x_dir = U
        y_dir = V
        z_dir = np.ones_like(U)

        norm = np.sqrt(x_dir ** 2 + y_dir ** 2 + z_dir ** 2)
        x_dir /= norm
        y_dir /= norm
        z_dir /= norm

        yaw_rad = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        x1 = x_dir * cos_yaw + z_dir * sin_yaw
        y1 = y_dir
        z1 = -x_dir * sin_yaw + z_dir * cos_yaw

        pitch_rad = math.radians(pitch_deg)
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        x2 = x1
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch

        theta = np.arctan2(x2, z2)
        phi = np.arcsin(np.clip(y2, -1.0, 1.0))

        px = ((theta / (2.0 * math.pi)) + 0.5) * equi_w
        py = (0.5 - (phi / math.pi)) * equi_h

        px = np.clip(px, 0, equi_w - 1).astype(np.int32)
        py = np.clip(py, 0, equi_h - 1).astype(np.int32)

        return equi_np[py, px]

    def view_panorama_pro(self, image, layout, create_video_frames, resolution, ratio, custom_ratio,
                          framerate, direction, num_loops, zoom, speed, depth_map=None):
        out_w, out_h = self._get_export_dims(resolution, ratio, custom_ratio)

        img_tensor = image[0]
        img_np = (255.0 * img_tensor.cpu().numpy()).astype(np.uint8)

        results = []
        pano_img = Image.fromarray(img_np)
        filename = "star_pano_pro_temp.jpg"
        file_path = os.path.join(self.output_dir, filename)
        pano_img.save(file_path, quality=90)
        results.append({
            "filename": filename,
            "type": self.type,
            "layout": layout,
            "subfolder": "",
            "export_width": out_w,
            "export_height": out_h,
            "create_video_frames": create_video_frames,
        })

        if depth_map is not None:
            depth_tensor = depth_map[0]
            d = (255.0 * depth_tensor.cpu().numpy()).astype(np.uint8)
            depth_img = Image.fromarray(d)
            depth_filename = "star_pano_pro_depth.jpg"
            depth_path = os.path.join(self.output_dir, depth_filename)
            depth_img.save(depth_path, quality=90)
            results[0]["depth_filename"] = depth_filename

        if not create_video_frames:
            placeholder = torch.zeros(1, out_h, out_w, 3)
            return {"result": (placeholder,), "ui": {"panoramas": results}}

        fov = 75.0 / max(zoom, 0.01)
        fov = max(10.0, min(120.0, fov))

        degrees_per_frame = speed / float(framerate)
        frames_per_loop = int(360.0 / degrees_per_frame) - 1
        frames_per_loop = max(1, frames_per_loop)

        if layout == "SBS":
            half_w = img_np.shape[1] // 2
            equi_np = img_np[:, :half_w, :]
        elif layout == "Top/Bottom":
            half_h = img_np.shape[0] // 2
            equi_np = img_np[:half_h, :, :]
        else:
            equi_np = img_np

        dir_sign = 1.0 if direction == "Left to Right" else -1.0

        one_loop_frames = []
        for i in range(frames_per_loop):
            yaw = (dir_sign * i * degrees_per_frame) % 360.0
            frame_np = self._equirect_to_perspective(equi_np, yaw, 0.0, fov, out_w, out_h)
            frame_tensor = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
            one_loop_frames.append(frame_tensor)

        one_loop_batch = torch.stack(one_loop_frames, dim=0)

        if num_loops > 1:
            frames_batch = one_loop_batch.repeat(num_loops, 1, 1, 1)
        else:
            frames_batch = one_loop_batch

        total_frames = frames_batch.shape[0]
        results[0]["total_frames"] = total_frames
        results[0]["frames_per_loop"] = frames_per_loop

        return {"result": (frames_batch,), "ui": {"panoramas": results}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "StarPanoramaViewerPro": StarPanoramaViewerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPanoramaViewerPro": "⭐ Star 360 Parallax Viewer Pro",
}
