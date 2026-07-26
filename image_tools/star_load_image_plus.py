"""
⭐ Star Load Image+

Loads an image (input folder, output folder, upload or clipboard paste) and
outputs the image, its mask, and the full metadata dict for "⭐ Star Image
Loader Options".
"""

import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

from .metadata_utils import read_star_metadata

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _list_input_images():
    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(_IMAGE_EXTS)
        ]
    except FileNotFoundError:
        files = []
    return sorted(files)


def _list_output_images():
    """Images inside the output folder (recursive), with the [output] annotation."""
    output_dir = folder_paths.get_output_directory()
    found = []
    try:
        for root, _dirs, names in os.walk(output_dir):
            for name in names:
                if name.lower().endswith(_IMAGE_EXTS):
                    rel = os.path.relpath(os.path.join(root, name), output_dir)
                    found.append(rel.replace(os.sep, "/"))
    except FileNotFoundError:
        pass
    return sorted(found)


class StarLoadImagePlus:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_images() + [f"{f} [output]" for f in _list_output_images()]
        return {
            "required": {
                "image": (
                    files,
                    {
                        "image_upload": True,
                        "__type__": "STRING",
                        "tooltip": "Image from the input folder, or from the output folder (entries ending with [output]), or upload / paste from clipboard.",
                    },
                ),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the mask extracted from the alpha channel (useful for inpaint vs. keep-area workflows).",
                    },
                ),
            }
        }

    CATEGORY = "⭐StarNodes/IO"
    RETURN_TYPES = ("IMAGE", "MASK", "STAR_METADATA")
    RETURN_NAMES = (
        "image",
        "mask",
        "metadata",
    )
    OUTPUT_TOOLTIPS = (
        "Loaded image.",
        "Alpha channel as mask (flip it with invert_mask).",
        "All metadata found in the image - connect to ⭐ Star Image Loader Options.",
    )
    FUNCTION = "load_image"
    DESCRIPTION = (
        "Load an image from the input or output folder. Outputs the image, its "
        "mask, and the full metadata for ⭐ Star Image Loader Options. Works with "
        "PNG, JPG and WEBP saved by ⭐ Star Save Image+."
    )

    def load_image(self, image, invert_mask=False):
        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        metadata = read_star_metadata(img)

        image_rgb = img.convert("RGB")
        image_arr = np.array(image_rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros(
                (image_tensor.shape[1], image_tensor.shape[2]),
                dtype=torch.float32,
                device="cpu",
            )

        if invert_mask:
            mask = 1.0 - mask

        img.close()

        entries = [[str(k), v if isinstance(v, str) else str(v)] for k, v in metadata.items()]
        return {
            "ui": {"star_metadata": [entries]},
            "result": (image_tensor, mask, metadata),
        }

    @classmethod
    def IS_CHANGED(cls, image, invert_mask=False, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            mtime = os.path.getmtime(image_path)
            size = os.path.getsize(image_path)
            return f"{mtime}-{size}-{invert_mask}"
        except Exception:
            return f"{image}-{invert_mask}"

    @classmethod
    def VALIDATE_INPUTS(cls, image, invert_mask=False, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarLoadImagePlus": StarLoadImagePlus,
    "⭐ Star Load Image+": StarLoadImagePlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLoadImagePlus": "⭐ Star Load Image+",
    "⭐ Star Load Image+": "⭐ Star Load Image+",
}
