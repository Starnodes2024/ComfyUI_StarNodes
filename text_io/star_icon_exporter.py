import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import folder_paths
import torch


def tensor_to_pil(img_tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] (float 0-1) to a PIL RGBA image (first batch).
    Adds opaque alpha if not present.
    """
    # IMAGE is expected as torch tensor, but we avoid importing torch to keep it light.
    # Use available tensor methods defensively.
    b, h, w, c = int(img_tensor.shape[0]), int(img_tensor.shape[1]), int(img_tensor.shape[2]), int(img_tensor.shape[3])
    # Take first in batch
    t = img_tensor[0]
    # Ensure CPU numpy
    if hasattr(t, 'detach'):
        t = t.detach()
    if hasattr(t, 'cpu'):
        t = t.cpu()
    np_img = np.clip(np.asarray(t), 0.0, 1.0)
    np_img = (np_img * 255.0).astype(np.uint8)  # H W C
    if c == 4:
        pil = Image.fromarray(np_img, mode='RGBA')
    else:
        pil = Image.fromarray(np_img, mode='RGB').convert('RGBA')
    return pil


def pil_to_tensor(img: Image.Image):
    """Convert a PIL RGBA/RGB image to a ComfyUI IMAGE tensor [1,H,W,C] float32 0-1."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    arr = np.asarray(img).astype(np.float32) / 255.0  # H W 4
    t = torch.from_numpy(arr)[None, ...]  # 1 H W 4
    return t


def parse_extra_sizes(extra: str) -> List[int]:
    out: List[int] = []
    if not extra:
        return out
    for part in extra.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
            if 8 <= val <= 1024:
                out.append(val)
        except ValueError:
            pass
    # dedupe and sort
    return sorted(list({*out}))


class StarIconExporter:
    """
    Star Icon Exporter
    - Takes an IMAGE input and a save name
    - Exports resized PNGs at standard icon sizes
    - Packs an .ico file containing the sizes
    - Optional extra sizes via comma-separated input
    - Optional 256-color quantization for PNGs and ICO
    """

    CATEGORY = "⭐StarNodes/Image And Latent"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("ico_path", "preview_batch")
    FUNCTION = "export_icons"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image input (first image in batch will be used)."}),
                "save_name": ("STRING", {"default": "icon", "tooltip": "Base filename without extension."}),
            },
            "optional": {
                "quantize_to_256": ("BOOLEAN", {"default": True, "tooltip": "Quantize to 256 colors for smaller files."}),
                "extra_sizes": ("STRING", {"default": "", "tooltip": "Comma-separated extra sizes, e.g. 64,512"}),
                "subfolder": ("STRING", {"default": "Icons", "tooltip": "Optional output subfolder under ComfyUI output."}),
                "shape": (["none", "square", "circle", "rounded"], {"default": "none", "tooltip": "Apply icon shape. 'square' fills background, 'circle'/'rounded' keep transparent background."}),
                "rounded_radius_percent": ("INT", {"default": 20, "min": 0, "max": 50, "tooltip": "Corner radius as percent of icon size (for rounded)."}),
                "background_color": ("STRING", {"default": "#FFFFFF", "tooltip": "HEX color for square background (e.g. #000000)."}),
                "padding_percent": ("INT", {"default": 0, "min": 0, "max": 40, "tooltip": "Uniform padding as percent of icon size."}),
                "auto_trim": ("BOOLEAN", {"default": True, "tooltip": "Auto-trim transparent edges before shaping."}),
                "stroke_enabled": ("BOOLEAN", {"default": False, "tooltip": "Draw an outline around the shape."}),
                "stroke_color": ("STRING", {"default": "#000000", "tooltip": "Outline color (HEX)."}),
                "stroke_width_percent": ("INT", {"default": 6, "min": 0, "max": 20, "tooltip": "Outline width as percent of icon size."}),
                "shadow_enabled": ("BOOLEAN", {"default": False, "tooltip": "Add a drop shadow behind the shape."}),
                "shadow_color": ("STRING", {"default": "#000000", "tooltip": "Shadow color (HEX)."}),
                "shadow_offset_px": ("INT", {"default": 2, "min": -20, "max": 20, "tooltip": "Shadow offset in pixels."}),
                "shadow_blur_px": ("INT", {"default": 4, "min": 0, "max": 32, "tooltip": "Shadow blur radius in pixels."}),
                "preset": (["standard", "windows", "android", "ios", "web_favicon"], {"default": "standard", "tooltip": "Preset size packs to include."}),
                "naming_style": (["increment", "underscore_increment", "timestamp"], {"default": "increment", "tooltip": "How to avoid overwriting existing files."}),
                "export_web_favicons": ("BOOLEAN", {"default": False, "tooltip": "Also export common web favicon PNG sizes."}),
                "preview_size": ("INT", {"default": 256, "min": 8, "max": 1024, "tooltip": "Preview size to show in node (closest available will be chosen)."}),
            },
        }

    @staticmethod
    def _build_sizes(extra_sizes: str) -> List[int]:
        base = [16, 32, 48, 128, 256]
        extra = parse_extra_sizes(extra_sizes)
        sizes = sorted(list({*base, *extra}))
        return sizes

    @staticmethod
    def _preset_sizes(preset: str) -> List[int]:
        presets = {
            "standard": [16, 32, 48, 128, 256],
            "windows": [16, 24, 32, 48, 64, 128, 256],
            "android": [48, 72, 96, 144, 192, 512],
            "ios": [60, 76, 120, 152, 167, 180, 1024],
            "web_favicon": [16, 32, 48, 96, 180, 192, 512],
        }
        return presets.get(preset, [16, 32, 48, 128, 256])

    @staticmethod
    def _pick_preview_size(available: List[int], desired: int) -> int:
        if not available:
            return desired
        if desired in available:
            return desired
        # choose closest size
        return min(available, key=lambda s: abs(s - desired))

    @staticmethod
    def _ensure_subfolder(base_dir: str, subfolder: str) -> str:
        if subfolder:
            out_dir = os.path.join(base_dir, subfolder)
        else:
            out_dir = base_dir
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _save_png(img_rgba: Image.Image, path: str, size: int, quantize: bool):
        im = img_rgba.resize((size, size), resample=Image.LANCZOS)
        if quantize:
            # Convert to paletted 256 colors while preserving alpha via RGBA -> P conversion
            im = im.convert("P", palette=Image.ADAPTIVE, colors=256)
        im.save(path, format="PNG")

    @staticmethod
    def _save_ico(img_rgba: Image.Image, path: str, sizes: List[int], quantize: bool):
        base = img_rgba
        if quantize:
            base = base.convert("P", palette=Image.ADAPTIVE, colors=256).convert("RGBA")
        # Pillow will generate downscaled sizes internally when passing sizes
        base.save(path, format="ICO", sizes=[(s, s) for s in sizes])

    @staticmethod
    def _unique_base_name(ico_directory: str, png_directory: str, base_name: str, sizes: List[int], style: str = "increment") -> str:
        """
        Find a non-conflicting base name across ICO dir and PNG dir.
        Rules: icon.ico, icon2.ico, icon3.ico ...
        Also checks PNGs like icon_16.png, icon2_16.png, etc. in the png subfolder.
        """
        suffix_num = 0  # 0 => no suffix, then 2,3,... (or _2,_3 for underscore_increment)
        while True:
            if suffix_num == 0:
                candidate = base_name
            else:
                if style == "underscore_increment":
                    candidate = f"{base_name}_{suffix_num}"
                else:
                    candidate = f"{base_name}{suffix_num}"
            ico_exists = os.path.exists(os.path.join(ico_directory, f"{candidate}.ico"))
            png_exists = any(
                os.path.exists(os.path.join(png_directory, f"{candidate}_{s}.png")) for s in sizes
            )
            if not ico_exists and not png_exists:
                return candidate
            # increment: skip 1 to match icon, icon2, icon3
            suffix_num = 2 if suffix_num == 0 else suffix_num + 1

    @staticmethod
    def _center_square(img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))

    @staticmethod
    def _parse_hex_color(hex_str: str) -> Tuple[int, int, int, int]:
        s = (hex_str or "").strip()
        if s.startswith('#'):
            s = s[1:]
        if len(s) == 3:
            s = ''.join(ch*2 for ch in s)
        if len(s) != 6:
            return (255, 255, 255, 255)
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b, 255)
        except ValueError:
            return (255, 255, 255, 255)

    @staticmethod
    def _shape_mask(size: int, shape: str, rounded_radius_percent: int) -> Image.Image:
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        if shape == "circle":
            draw.ellipse((0, 0, size - 1, size - 1), fill=255)
        elif shape == "rounded":
            r = max(0, min(50, int(rounded_radius_percent)))
            rad = int(size * r / 100)
            # rounded rectangle
            draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=rad, fill=255)
        else:
            # full square
            draw.rectangle((0, 0, size - 1, size - 1), fill=255)
        return mask

    def _render_icon(self, base_rgba: Image.Image, size: int, shape: str, rounded_radius_percent: int, background_color: str) -> Image.Image:
        # optional auto-trim transparent bounds to keep content centered and tight
        src = base_rgba
        if hasattr(self, "_auto_trim_flag") and self._auto_trim_flag:
            src = self._auto_trim_rgba(src)
        # prepare square crop then resize
        img = self._center_square(src)
        # apply padding by scaling down
        pad = max(0, min(40, int(getattr(self, "_padding_percent", 0))))
        if pad > 0:
            inner = int(round(size * (100 - 2 * pad) / 100))
            inner = max(1, min(size, inner))
            img = img.resize((inner, inner), Image.LANCZOS)
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ox = (size - inner) // 2
            oy = (size - inner) // 2
            canvas.paste(img, (ox, oy))
            img = canvas
        else:
            img = img.resize((size, size), Image.LANCZOS)
        if shape == "none":
            return img
        if shape == "square":
            # composite onto solid background to remove transparency
            bg = Image.new("RGBA", (size, size), self._parse_hex_color(background_color))
            bg.paste(img, (0, 0), img)
            return bg
        # circle or rounded: create transparency outside
        mask = self._shape_mask(size, shape, rounded_radius_percent)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        return out

    @staticmethod
    def _auto_trim_rgba(img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bbox = img.split()[3].point(lambda a: 255 if a > 0 else 0).getbbox()
        if bbox:
            return img.crop(bbox)
        return img

    @staticmethod
    def _apply_shadow(base: Image.Image, shape_mask: Image.Image, color: Tuple[int, int, int, int], offset: Tuple[int, int], blur: int) -> Image.Image:
        w, h = base.size
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # create colored shadow from mask
        sh_mask = shape_mask.copy()
        if blur > 0:
            sh_mask = sh_mask.filter(ImageFilter.GaussianBlur(radius=blur))
        sh_layer = Image.new("RGBA", (w, h), color)
        shadow.paste(sh_layer, offset, sh_mask)
        out = Image.alpha_composite(shadow, base)
        return out

    @staticmethod
    def _apply_stroke(base: Image.Image, shape: str, size: int, rounded_radius_percent: int, color: Tuple[int, int, int, int], width_percent: int) -> Image.Image:
        w = max(0, min(20, int(width_percent)))
        px = max(1, int(size * w / 100)) if w > 0 else 0
        if px == 0:
            return base
        stroke_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(stroke_layer)
        if shape == "circle":
            draw.ellipse((px//2, px//2, size - 1 - px//2, size - 1 - px//2), outline=color, width=px)
        elif shape == "rounded":
            r = max(0, min(50, int(rounded_radius_percent)))
            rad = int(size * r / 100)
            draw.rounded_rectangle((px//2, px//2, size - 1 - px//2, size - 1 - px//2), radius=rad, outline=color, width=px)
        else:
            # square/none
            draw.rectangle((px//2, px//2, size - 1 - px//2, size - 1 - px//2), outline=color, width=px)
        return Image.alpha_composite(base, stroke_layer)

    @staticmethod
    def _draw_size_label(img: Image.Image, text: str) -> Image.Image:
        # Draw a semi-transparent dark bar and centered white text at bottom
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        # slightly taller bar for bigger text
        bar_h = max(14, int(h * 0.18))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # bar
        draw.rectangle((0, h - bar_h, w, h), fill=(0, 0, 0, 170))
        # text
        used_default_font = False
        try:
            from PIL import ImageFont
            font = None
            target_px = max(14, int(bar_h * 0.80))
            # try multiple bold TTF locations
            candidates = [
                "DejaVuSans-Bold.ttf",
                # Windows common fonts
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arialbd.ttf"),
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeuib.ttf"),
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "seguisb.ttf"),
                # Linux common
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            ]
            try:
                import PIL
                pil_dir = os.path.dirname(PIL.__file__)
                candidates.append(os.path.join(pil_dir, "fonts", "DejaVuSans-Bold.ttf"))
            except Exception:
                pass
            for fp in candidates:
                try:
                    if fp and os.path.exists(fp) or (isinstance(fp, str) and os.path.basename(fp) == fp):
                        font = ImageFont.truetype(fp, target_px)
                        break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()
                used_default_font = True
        except Exception:
            font = None
            used_default_font = True
        # Pillow >= 8 provides textbbox; textsize may not exist in newer versions
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            # Fallback to font.getsize if available, else approximate
            try:
                tw, th = font.getsize(text) if font else (len(text) * 6, 11)
            except Exception:
                tw, th = (len(text) * 6, 11)
        tx = (w - tw) // 2
        ty = h - bar_h + (bar_h - th) // 2
        # If only bitmap default font is available, upscale a text tile for larger appearance
        if used_default_font and font is not None:
            # render small text onto a tiny tile, then scale up to match ~target_px height
            tmp_w = max(1, tw)
            tmp_h = max(1, th)
            tile = Image.new("RGBA", (tmp_w+4, tmp_h+4), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tile)
            # faux bold outline
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)):
                tdraw.text((2+dx, 2+dy), text, fill=(0,0,0,220), font=font)
            tdraw.text((2, 2), text, fill=(255,255,255,255), font=font)
            # compute scale
            target_h = max(12, int(bar_h * 0.7))
            scale = max(1, int(round(target_h / tile.size[1])))
            scaled = tile.resize((tile.size[0]*scale, tile.size[1]*scale), resample=Image.LANCZOS)
            # paste centered
            sx, sy = scaled.size
            cx = (w - sx) // 2
            cy = h - bar_h + (bar_h - sy) // 2
            overlay.alpha_composite(scaled, (cx, cy))
        else:
            # draw bold with stroke if available, else multi-pass faux bold
            try:
                sw = 3 if max(w, h) >= 200 else 2
                draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font, stroke_width=sw, stroke_fill=(0, 0, 0, 220))
            except TypeError:
                for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)):
                    draw.text((tx+dx, ty+dy), text, fill=(0, 0, 0, 200), font=font)
                draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)
        return Image.alpha_composite(img, overlay)

    def export_icons(self, images, save_name: str = "icon", quantize_to_256: bool = True, extra_sizes: str = "", subfolder: str = "Icons", shape: str = "none", rounded_radius_percent: int = 20, background_color: str = "#FFFFFF", padding_percent: int = 0, auto_trim: bool = True, stroke_enabled: bool = False, stroke_color: str = "#000000", stroke_width_percent: int = 6, shadow_enabled: bool = False, shadow_color: str = "#000000", shadow_offset_px: int = 2, shadow_blur_px: int = 4, preset: str = "standard", naming_style: str = "increment", export_web_favicons: bool = False, preview_size: int = 256):
        output_dir = folder_paths.get_output_directory()
        out_dir = self._ensure_subfolder(output_dir, subfolder)
        # If a preset is selected, add a preset subfolder for clearer organization
        preset_name = (preset or "standard").strip()
        if preset_name and preset_name != "standard":
            out_dir = self._ensure_subfolder(out_dir, preset_name)
        png_dir = os.path.join(out_dir, "png")
        os.makedirs(png_dir, exist_ok=True)

        # Convert tensor to base RGBA image
        pil_rgba = tensor_to_pil(images)

        # Build sizes list from preset + custom extras
        sizes = sorted(list({*self._preset_sizes(preset), *self._build_sizes(extra_sizes)}))

        # Choose a unique base name to avoid overwriting existing files
        base_name = self._unique_base_name(out_dir, png_dir, save_name, sizes, naming_style) if naming_style in ("increment", "underscore_increment") else self._timestamp_base_name(out_dir, save_name)

        # Persist flags for render helpers
        self._padding_percent = padding_percent
        self._auto_trim_flag = bool(auto_trim)

        # Save PNGs per size
        saved_pngs: List[Tuple[str, int]] = []
        shaped_by_size: dict[int, Image.Image] = {}
        for s in sizes:
            png_path = os.path.join(png_dir, f"{base_name}_{s}.png")
            shaped = self._render_icon(pil_rgba, s, shape, rounded_radius_percent, background_color)
            # Apply shadow beneath if requested
            if shadow_enabled and shape != "none":
                mask = self._shape_mask(s, shape, rounded_radius_percent)
                shaped = self._apply_shadow(shaped, mask, self._parse_hex_color(shadow_color), (shadow_offset_px, shadow_offset_px), max(0, int(shadow_blur_px)))
            # Apply stroke on top if requested
            if stroke_enabled and shape != "none":
                shaped = self._apply_stroke(shaped, shape, s, rounded_radius_percent, self._parse_hex_color(stroke_color), stroke_width_percent)
            # keep for preview batch assembly
            shaped_by_size[s] = shaped
            # Preserve transparency for circle/rounded by disabling quantization for PNGs
            quantize_png = quantize_to_256 and shape not in ("circle", "rounded")
            self._save_png(shaped, png_path, s, quantize_png)
            saved_pngs.append((png_path, s))

        # Save ICO with all sizes using largest shaped image as base
        ico_path = os.path.join(out_dir, f"{base_name}.ico")
        largest = max(sizes)
        shaped_largest = self._render_icon(pil_rgba, largest, shape, rounded_radius_percent, background_color)
        if shadow_enabled and shape != "none":
            mask_l = self._shape_mask(largest, shape, rounded_radius_percent)
            shaped_largest = self._apply_shadow(shaped_largest, mask_l, self._parse_hex_color(shadow_color), (shadow_offset_px, shadow_offset_px), max(0, int(shadow_blur_px)))
        if stroke_enabled and shape != "none":
            shaped_largest = self._apply_stroke(shaped_largest, shape, largest, rounded_radius_percent, self._parse_hex_color(stroke_color), stroke_width_percent)
        self._save_ico(shaped_largest, ico_path, sizes, quantize_to_256)

        # Optional: export web favicon PNG set regardless of preset
        if export_web_favicons:
            web_sizes = [16, 32, 48, 96, 180, 192, 512]
            for s in web_sizes:
                if s in sizes:
                    continue
                png_path = os.path.join(png_dir, f"{base_name}_{s}.png")
                shaped = self._render_icon(pil_rgba, s, shape, rounded_radius_percent, background_color)
                if shadow_enabled and shape != "none":
                    mask = self._shape_mask(s, shape, rounded_radius_percent)
                    shaped = self._apply_shadow(shaped, mask, self._parse_hex_color(shadow_color), (shadow_offset_px, shadow_offset_px), max(0, int(shadow_blur_px)))
                if stroke_enabled and shape != "none":
                    shaped = self._apply_stroke(shaped, shape, s, rounded_radius_percent, self._parse_hex_color(stroke_color), stroke_width_percent)
                quantize_png = quantize_to_256 and shape not in ("circle", "rounded")
                self._save_png(shaped, png_path, s, quantize_png)

        # Build a single UI preview for the chosen preview size
        ui_results = []
        # compute closest available preview size
        preview_s = self._pick_preview_size(sizes, int(preview_size))
        preview_file = os.path.join(png_dir, f"{base_name}_{preview_s}.png")
        # Report subfolder path for preview
        if subfolder:
            png_subfolder = os.path.join(subfolder, preset_name) if preset_name and preset_name != "standard" else subfolder
            png_subfolder = os.path.join(png_subfolder, "png")
        else:
            png_subfolder = os.path.join(preset_name, "png") if preset_name and preset_name != "standard" else "png"
        ui_results.append({
            "filename": os.path.basename(preview_file),
            "subfolder": png_subfolder,
            "type": "output"
        })

        # Build preview batch: center each shaped icon on largest canvas and draw size label
        largest = max(sizes)
        batch_tensors = []
        for s in sizes:
            shaped_img = shaped_by_size.get(s)
            if shaped_img is None:
                shaped_img = self._render_icon(pil_rgba, s, shape, rounded_radius_percent, background_color)
            canvas = Image.new("RGBA", (largest, largest), (0, 0, 0, 0))
            ox = (largest - s) // 2
            oy = (largest - s) // 2
            canvas.paste(shaped_img, (ox, oy), shaped_img)
            labeled = self._draw_size_label(canvas, f"{s}x{s}")
            batch_tensors.append(pil_to_tensor(labeled))
        # concatenate into batch [B,H,W,C]
        if len(batch_tensors) > 1:
            preview_batch = torch.cat(batch_tensors, dim=0)
        else:
            preview_batch = batch_tensors[0]

        return {"ui": {"images": ui_results}, "result": (ico_path, preview_batch)}

    @staticmethod
    def _timestamp_base_name(directory: str, base_name: str) -> str:
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        candidate = f"{base_name}_{ts}"
        # Ensure no conflicts
        idx = 2
        while os.path.exists(os.path.join(directory, f"{candidate}.ico")):
            candidate = f"{base_name}_{ts}_{idx}"
            idx += 1
        return candidate


NODE_CLASS_MAPPINGS = {
    "StarIconExporter": StarIconExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarIconExporter": "⭐ Star Icon Exporter",
}
