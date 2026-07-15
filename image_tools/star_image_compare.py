import folder_paths
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch


class StarImageCompare:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE", {"forceInput": True}),
                "image_2": ("IMAGE", {"forceInput": True}),
                "compare_position": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Initial slider position. 0.0 = Image 2, 1.0 = Image 1."
                }),
                "caption_image1": ("STRING", {"default": ""}),
                "caption_image2": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("concat_image",)
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Interactive before/after image comparison with a draggable slider and a concatenated output image."

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor to a PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        np_image = (tensor.cpu().numpy() * 255.0)
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def compare_images(self, image_1, image_2, compare_position=0.5, caption_image1="", caption_image2=""):
        ui_data = {"image1": None, "image2": None, "compare_position": compare_position}

        if image_1 is not None and image_2 is not None:
            def save_tensor(tensor):
                img = self.tensor_to_pil(tensor)
                filename = f"star_compare_{uuid.uuid4().hex}.png"
                file_path = os.path.join(self.output_dir, filename)
                img.save(file_path, compress_level=3)
                return {
                    "filename": filename,
                    "type": self.type,
                    "subfolder": ""
                }

            meta1 = save_tensor(image_1)
            meta2 = save_tensor(image_2)
            ui_data = {"image1": meta1, "image2": meta2, "compare_position": compare_position}

        concat_tensor = self._create_concat_image(image_1, image_2, caption_image1, caption_image2)

        return {
            "ui": {"star_image_compare": [ui_data]},
            "result": (concat_tensor,)
        }

    def _pil_to_tensor(self, pil: Image.Image) -> torch.Tensor:
        np_image = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _get_font(self, size=18):
        font_names = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "segoeui.ttf", "Verdana.ttf"]
        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _draw_caption_bar(self, img: Image.Image, caption: str, draw_bar: bool) -> Image.Image:
        if not draw_bar:
            return img
        w, h = img.size
        bar_h = max(24, min(140, int(h * 0.08)))
        font_size = max(16, int(bar_h * 0.65))
        out = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
        out.paste(img, (0, 0))
        text = (caption or "").strip()
        if text:
            draw = ImageDraw.Draw(out)
            font = self._get_font(font_size)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                x = (w - tw) / 2 - bbox[0]
                y = h + (bar_h - th) / 2 - bbox[1]
            except Exception:
                try:
                    tw, th = draw.textsize(text, font=font)
                except Exception:
                    tw, th = w, bar_h
                x = (w - tw) / 2
                y = h + (bar_h - th) / 2
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return out

    def _concat_two_images(self, img1: Image.Image, img2: Image.Image, cap1: str, cap2: str) -> Image.Image:
        def is_portrait(img):
            return img.height > img.width

        p1 = is_portrait(img1)
        p2 = is_portrait(img2)

        has_cap = bool((cap1 or "").strip()) or bool((cap2 or "").strip())

        if p1 and p2:
            # Both portrait -> side by side, match heights of the smaller
            target_h = min(img1.height, img2.height)
            if img1.height > target_h:
                img1 = img1.resize((max(1, int(img1.width * target_h / img1.height)), target_h), Image.LANCZOS)
            if img2.height > target_h:
                img2 = img2.resize((max(1, int(img2.width * target_h / img2.height)), target_h), Image.LANCZOS)

            seg1 = self._draw_caption_bar(img1, cap1, has_cap)
            seg2 = self._draw_caption_bar(img2, cap2, has_cap)

            total_w = seg1.width + seg2.width
            total_h = max(seg1.height, seg2.height)
            out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            out.paste(seg1, (0, 0))
            out.paste(seg2, (seg1.width, 0))
        else:
            # Landscape or mixed -> stacked vertically, match widths of the smaller
            target_w = min(img1.width, img2.width)
            if img1.width > target_w:
                img1 = img1.resize((target_w, max(1, int(img1.height * target_w / img1.width))), Image.LANCZOS)
            if img2.width > target_w:
                img2 = img2.resize((target_w, max(1, int(img2.height * target_w / img2.width))), Image.LANCZOS)

            seg1 = self._draw_caption_bar(img1, cap1, has_cap)
            seg2 = self._draw_caption_bar(img2, cap2, has_cap)

            total_w = max(seg1.width, seg2.width)
            total_h = seg1.height + seg2.height
            out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            out.paste(seg1, (0, 0))
            out.paste(seg2, (0, seg1.height))

        return out

    def _create_concat_image(self, image_1, image_2, caption_image1, caption_image2):
        if image_1 is None and image_2 is None:
            pil = Image.new("RGB", (64, 64), (0, 0, 0))
        elif image_1 is None:
            img = self.tensor_to_pil(image_2)
            pil = self._draw_caption_bar(img, caption_image2, bool((caption_image2 or "").strip()))
        elif image_2 is None:
            img = self.tensor_to_pil(image_1)
            pil = self._draw_caption_bar(img, caption_image1, bool((caption_image1 or "").strip()))
        else:
            pil = self._concat_two_images(
                self.tensor_to_pil(image_1),
                self.tensor_to_pil(image_2),
                caption_image1 or "",
                caption_image2 or ""
            )
        return self._pil_to_tensor(pil)


NODE_CLASS_MAPPINGS = {
    "StarImageCompare": StarImageCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageCompare": "⭐ Star Image Compare",
}
