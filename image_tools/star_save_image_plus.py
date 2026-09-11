"""
⭐ Star Save Image+

Drop-in replacement for the classic ⭐ Star Save Image+ node:
  - mode: "save" or "preview" (driven by the DOM segmented control)
  - formats: any combination of png / jpg / webp / psd (DOM multi-select chips)
  - optional MASK input, embedded as alpha channel (PNG/WEBP) or PSD layer mask
  - metadata: ONLY the 5 custom StarMetaData fields, coming from the optional
    "⭐ Star Metadata Saver Option" node. No workflow/prompt is embedded anymore.
    (NOTE: workflow/prompt IS now embedded for PNG so drag-and-drop works.)

Old workflows made with the original ⭐ Star Save Image+ keep working.
"""

import json
import os
import re
import struct
import zlib
from datetime import datetime

import numpy as np
from PIL import Image

import folder_paths
from .metadata_utils import build_exif_bytes, build_png_info

try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    from psd_tools.constants import ChannelID, Compression
    from psd_tools.psd.layer_and_mask import MaskData, MaskFlags, ChannelInfo, ChannelData
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

_PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")


def _load_presets():
    try:
        with open(_PRESETS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("presets", [])
        if isinstance(data, list):
            return [str(item) for item in data if str(item).strip()]
    except Exception:
        pass
    return []


def _clean(value):
    return "" if value is None else str(value).strip()


def build_prefix(
    preset_folder="None",
    date_folder=True,
    date_folder_position="first",
    custom_folder="",
    custom_subfolder="",
    date_in_filename="Off",
    filename="ComfyUI",
    add_timestamp=False,
    separator="_",
):
    folders = []
    today = datetime.now().strftime("%Y-%m-%d")

    preset_folder = _clean(preset_folder)
    custom_folder = _clean(custom_folder)
    custom_subfolder = _clean(custom_subfolder)
    date_folder_position = _clean(date_folder_position) or "first"
    date_in_filename = _clean(date_in_filename) or "Off"
    filename = _clean(filename) or "ComfyUI"
    separator = "" if separator is None else str(separator)

    if date_folder and date_folder_position == "first":
        folders.append(today)

    base_folder = preset_folder if preset_folder and preset_folder.lower() != "none" else custom_folder
    if base_folder:
        folders.append(base_folder.strip("/").strip())

    if custom_subfolder:
        folders.append(custom_subfolder.strip("/").strip())

    if date_folder and date_folder_position == "subfolder":
        folders.append(today)

    name = filename

    if date_in_filename == "prefix":
        name = f"{today}{separator}{name}"
    elif date_in_filename == "suffix":
        name = f"{name}{separator}{today}"

    if add_timestamp:
        name = f"{name}{separator}{datetime.now().strftime('%H%M%S')}"

    folders = [f for f in folders if f]
    return "/".join(folders + [name]) if folders else name


def _parse_formats(raw):
    parts = None

    if isinstance(raw, (list, tuple, set)):
        parts = [str(x) for x in raw]
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                data = json.loads(stripped)
                if isinstance(data, list):
                    parts = [str(x) for x in data]
            except Exception:
                parts = None

        if parts is None:
            parts = re.split(r"\s*,\s*", stripped)
    else:
        parts = [str(raw or "")]

    formats = []
    for part in parts:
        part = part.strip().lower()
        if part in ("png", "jpg", "jpeg", "webp", "psd"):
            fmt = "jpg" if part == "jpeg" else part
            if fmt not in formats:
                formats.append(fmt)

    return formats or ["png"]


def _png_chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def save_png_16bit(path, arr_uint16, compress_level=4, text_metadata=None):
    """Write a 16-bit-per-channel PNG (grayscale/RGB/RGBA) without relying on
    Pillow, which does not support saving multi-channel 16-bit PNGs.

    arr_uint16: numpy array, dtype uint16, shape (H, W) / (H, W, 3) / (H, W, 4).
    """
    if arr_uint16.ndim == 2:
        height, width = arr_uint16.shape
        channels = 1
    else:
        height, width, channels = arr_uint16.shape

    color_type = {1: 0, 3: 2, 4: 6}.get(channels)
    if color_type is None:
        raise ValueError(f"Unsupported channel count for 16-bit PNG: {channels}")

    # PNG requires big-endian sample order.
    arr_be = np.ascontiguousarray(arr_uint16.astype(">u2"))
    row_bytes = arr_be.reshape(height, -1).tobytes()
    stride = width * channels * 2

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None) per scanline
        raw += row_bytes[y * stride:(y + 1) * stride]

    compressed = zlib.compress(bytes(raw), max(0, min(9, int(compress_level))))

    ihdr = struct.pack(">IIBBBBB", width, height, 16, color_type, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        if text_metadata:
            for key, value in text_metadata.items():
                key_b = str(key).encode("latin-1", errors="replace")[:79]
                val_b = str(value).encode("latin-1", errors="replace")
                f.write(_png_chunk(b"tEXt", key_b + b"\x00" + val_b))
        f.write(_png_chunk(b"IDAT", compressed))
        f.write(_png_chunk(b"IEND", b""))


class StarSaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = ["None"] + _load_presets()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "preset_folder": (
                    preset_options,
                    {
                        "default": "None",
                        "tooltip": "Preset folder from presets.json (overrides custom folder).",
                    },
                ),
                "date_folder": ("BOOLEAN", {"default": True, "tooltip": "Create a folder with today's date."}),
                "date_folder_position": (["first", "subfolder"], {"default": "first", "tooltip": "Where to place the date folder: 'first' = top-level, 'subfolder' = inside the preset/custom folder."}),
                "custom_folder": ("STRING", {"default": "", "multiline": False, "tooltip": "Custom output folder name (used when preset_folder is None)."}),
                "custom_subfolder": ("STRING", {"default": "", "multiline": False, "tooltip": "Additional subfolder inside the main folder."}),
                "date_in_filename": (["Off", "prefix", "suffix"], {"default": "Off", "tooltip": "Add today's date to the filename: 'prefix' = before the name, 'suffix' = after."}),
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False, "tooltip": "Base filename for saved images (a counter is appended automatically)."}),
                "add_timestamp": ("BOOLEAN", {"default": False, "tooltip": "Append HHMMSS time to the filename for unique, sortable names."}),
                "separator": ("STRING", {"default": "_", "multiline": False, "tooltip": "Separator between filename parts (date, name, timestamp, counter)."}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality (1-100)."}),
                "webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1, "tooltip": "WEBP quality (1-100)."}),
                "png_compress": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1, "tooltip": "PNG compression level (0-9)."}),
                "png_bit_depth": (["8bit", "16bit"], {"default": "8bit", "tooltip": "Bit depth for saved PNG files. 16bit preserves more precision from the source tensor (larger files, no EXIF, only tEXt metadata)."}),
                # Driven by the DOM widgets (hidden on the node):
                "mode": ("STRING", {"default": "save", "tooltip": "save or preview (controlled by the on-node buttons)."}),
                "formats": ("STRING", {"default": "png", "tooltip": "Comma separated formats: png,jpg,webp,psd (controlled by the on-node chips)."}),
            },
            "optional": {
                "options": ("STAR_SAVE_OPTIONS", {"tooltip": "Optional ⭐ Star Metadata Saver Option output (provides the 5 StarMetaData fields to embed)."}),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Optional mask - embedded as alpha channel (PNG/WEBP) so ⭐ Star Load Image+ can restore it. For JPG a separate _mask.png sidecar is written. For PSD it is embedded as a layer mask."
                    },
                ),
            },
            # ─── NEW: hidden inputs injected by ComfyUI ───
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = ("Save folder path (relative to the ComfyUI output directory, without filename).",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/IO"
    DESCRIPTION = (
        "Save images as PNG / JPG / WEBP / PSD (multi-select) or just preview them. "
        "Embeds up to 5 custom StarMetaData fields via ⭐ Star Metadata Saver Option. "
        "No workflow data is stored."
    )

    @staticmethod
    def _mask_for_index(mask, index, size):
        """Return an alpha PIL image (L mode) for batch item `index`.
        ComfyUI mask convention: 1.0 = masked area. We store it as alpha
        (alpha = 1 - mask) so the loader roundtrip restores it.
        """
        if mask is None:
            return None

        if mask.dim() == 3:
            entry = mask[index] if index < mask.shape[0] else mask[-1]
        else:
            entry = mask

        alpha = (1.0 - entry).cpu().numpy()
        alpha_img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")

        if alpha_img.size != size:
            resampling = getattr(Image, "Resampling", Image)
            alpha_img = alpha_img.resize(size, resampling.BILINEAR)

        return alpha_img

    @staticmethod
    def _mask_for_index_16(mask, index, size):
        """Same as _mask_for_index but returns a full-precision uint16 (H, W)
        alpha array, for embedding in 16-bit PNGs."""
        if mask is None:
            return None

        if mask.dim() == 3:
            entry = mask[index] if index < mask.shape[0] else mask[-1]
        else:
            entry = mask

        alpha = np.clip((1.0 - entry).cpu().numpy(), 0, 1).astype(np.float32)

        if (alpha.shape[1], alpha.shape[0]) != size:
            alpha_img = Image.fromarray(alpha, mode="F")
            resampling = getattr(Image, "Resampling", Image)
            alpha_img = alpha_img.resize(size, resampling.BILINEAR)
            alpha = np.array(alpha_img, dtype=np.float32)

        return np.clip(alpha * 65535.0, 0, 65535).astype(np.uint16)

    @staticmethod
    def _resolve_metadata(options):
        """Extract the 5 custom StarMetaData fields from the options node."""
        if not isinstance(options, dict):
            return {}
        option_metadata = options.get("metadata", {}) or {}
        metadata = {}
        for i in range(1, 6):
            value = option_metadata.get(f"StarMetaData {i}")
            if value not in (None, ""):
                metadata[f"StarMetaData {i}"] = value
        return metadata

    def save_images(
        self,
        images,
        preset_folder="None",
        date_folder=True,
        date_folder_position="first",
        custom_folder="",
        custom_subfolder="",
        date_in_filename="Off",
        filename="ComfyUI",
        add_timestamp=False,
        separator="_",
        jpg_quality=95,
        webp_quality=90,
        png_compress=4,
        png_bit_depth="8bit",
        mode="save",
        formats="png",
        options=None,
        mask=None,
        # ─── NEW ───
        prompt=None,
        extra_pnginfo=None,
    ):
        if images is None or len(images) == 0:
            return {
                "ui": {"images": []},
                "result": ("",),
            }

        preview = str(mode).strip().lower() == "preview"
        selected_formats = ["png"] if preview else _parse_formats(formats)

        if preview:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                "ComfyUI_temp_star", self.temp_dir, images[0].shape[1], images[0].shape[0]
            )
            node_type = "temp"
        else:
            filename_prefix = build_prefix(
                preset_folder=preset_folder,
                date_folder=date_folder,
                date_folder_position=date_folder_position,
                custom_folder=custom_folder,
                custom_subfolder=custom_subfolder,
                date_in_filename=date_in_filename,
                filename=filename,
                add_timestamp=add_timestamp,
                separator=separator,
            )
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
            node_type = "output"

        results = []
        saved_files = []

        save_png_16 = (not preview) and str(png_bit_depth).strip().lower() == "16bit"

        for batch_number, image in enumerate(images):
            tensor = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(tensor, 0, 255).astype(np.uint8))

            # Optional mask -> alpha channel (PNG / WEBP roundtrip with the loader).
            alpha_img = self._mask_for_index(mask, batch_number, img.size)
            if alpha_img is not None:
                img = img.convert("RGBA")
                img.putalpha(alpha_img)

            # 16-bit PNG needs the full-precision float source (Pillow can only
            # save 8-bit multi-channel PNGs), so build a separate uint16 array.
            arr16 = None
            if "png" in selected_formats and save_png_16:
                rgb16 = np.clip(image.cpu().numpy() * 65535.0, 0, 65535).astype(np.uint16)
                alpha16 = self._mask_for_index_16(mask, batch_number, img.size)
                if alpha16 is not None:
                    arr16 = np.dstack([rgb16, alpha16])
                else:
                    arr16 = rgb16

            metadata = {}
            exif_bytes = None
            png_info = None

            if not preview:
                metadata = self._resolve_metadata(options)

                # ─── NEW: build PNG info with workflow/prompt data ───
                from PIL.PngImagePlugin import PngInfo
                png_info = PngInfo()

                if prompt is not None:
                    png_info.add_text("prompt", json.dumps(prompt))

                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        png_info.add_text(key, json.dumps(value))

                # Add custom StarMetaData on top
                for key, value in metadata.items():
                    png_info.add_text(key, str(value))

                # EXIF for JPG/WEBP (custom metadata only, same as before)
                if metadata:
                    exif_bytes = build_exif_bytes(metadata)
                # ─── END NEW ───

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            first_file = None

            for fmt in selected_formats:
                file = f"{filename_with_batch_num}_{counter:05}_.{fmt}"
                path = os.path.join(full_output_folder, file)

                if fmt == "png":
                    if save_png_16 and arr16 is not None:
                        # ─── NEW: merge workflow data into 16-bit PNG text chunks ───
                        text_metadata = dict(metadata)
                        if prompt is not None:
                            text_metadata["prompt"] = json.dumps(prompt)
                        if extra_pnginfo is not None:
                            for key, value in extra_pnginfo.items():
                                text_metadata[key] = json.dumps(value)
                        # ─── END NEW ───
                        save_png_16bit(path, arr16, compress_level=int(png_compress), text_metadata=text_metadata)
                    else:
                        img.save(path, pnginfo=png_info, compress_level=int(png_compress))

                elif fmt == "jpg":
                    save_kwargs = {"quality": int(jpg_quality)}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes

                    img.convert("RGB").save(path, "JPEG", **save_kwargs)

                    # JPG has no alpha: write a sidecar mask file.
                    if alpha_img is not None:
                        mask_file = f"{filename_with_batch_num}_{counter:05}_mask.png"
                        raw_mask = Image.fromarray(255 - np.array(alpha_img), mode="L")
                        raw_mask.save(os.path.join(full_output_folder, mask_file))
                        saved_files.append(os.path.join(subfolder, mask_file) if subfolder else mask_file)

                elif fmt == "webp":
                    save_kwargs = {"quality": int(webp_quality), "method": 6}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes

                    img.save(path, "WEBP", **save_kwargs)

                elif fmt == "psd":
                    if not PSD_TOOLS_AVAILABLE:
                        raise RuntimeError(
                            "psd-tools is not installed. Run: pip install psd-tools"
                        )
                    rgb_img = img.convert("RGB")
                    psd = PSDImage.new(mode="RGB", size=rgb_img.size)
                    layer = PixelLayer.frompil(rgb_img, psd, "Layer 1")
                    psd.append(layer)

                    if alpha_img is not None:
                        raw_mask = Image.fromarray(255 - np.array(alpha_img), mode="L")
                        if raw_mask.mode != "L":
                            raw_mask = raw_mask.convert("L")
                        mask_data = MaskData(
                            top=0, left=0,
                            bottom=raw_mask.height, right=raw_mask.width,
                            background_color=0,
                            flags=MaskFlags(parameters_applied=True),
                        )
                        layer._record.mask_data = mask_data
                        channel_data = ChannelData(compression=Compression.RAW)
                        channel_data.set_data(
                            raw_mask.tobytes(),
                            raw_mask.width,
                            raw_mask.height,
                            8,
                            1,
                        )
                        layer._record.channel_info.append(
                            ChannelInfo(id=ChannelID.USER_LAYER_MASK, length=0)
                        )
                        layer._channels.append(channel_data)

                    psd.save(path)

                saved_files.append(os.path.join(subfolder, file) if subfolder else file)

                if first_file is None:
                    first_file = file

            # Show the first written format in the node preview.
            results.append({"filename": first_file, "subfolder": subfolder, "type": node_type})
            counter += 1

        return {
            "ui": {"images": results, "star_files": [saved_files]},
            "result": (subfolder,),
        }


NODE_CLASS_MAPPINGS = {
    "⭐ Star Save Image+": StarSaveImagePlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "⭐ Star Save Image+": "⭐ Star Save Image+",
}
