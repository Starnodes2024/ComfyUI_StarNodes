import os
import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths


class StarLoadImagePlus:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
        ]
        return {
            "required": {
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "__type__": "STRING",
                    },
                )
            }
        }

    CATEGORY = "⭐StarNodes/IO"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "image",
        "mask",
        "StarMetaData 1",
        "StarMetaData 2",
        "StarMetaData 3",
        "StarMetaData 4",
        "StarMetaData 5",
    )
    FUNCTION = "load_image"

    def _read_star_metadata(self, img: Image.Image):
        md = {}

        if hasattr(img, "text") and isinstance(getattr(img, "text"), dict):
            md.update(dict(img.text))

        if hasattr(img, "info") and isinstance(getattr(img, "info"), dict):
            for k, v in img.info.items():
                if isinstance(v, (str, bytes)):
                    if isinstance(v, bytes):
                        try:
                            v = v.decode("utf-8")
                        except Exception:
                            continue
                    md[k] = v

        # WebP EXIF style used by StarMetaInjector: "key:value" and "prompt:<json>"
        try:
            exif_data = img.getexif()
            if exif_data is not None:
                for _, exif_value in exif_data.items():
                    if isinstance(exif_value, str) and ":" in exif_value:
                        k, v = exif_value.split(":", 1)
                        if k and v:
                            md[k] = v
        except Exception:
            pass

        return (
            md.get("StarMetaData 1", ""),
            md.get("StarMetaData 2", ""),
            md.get("StarMetaData 3", ""),
            md.get("StarMetaData 4", ""),
            md.get("StarMetaData 5", ""),
        )

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)

        star_md = self._read_star_metadata(i)

        image_rgb = i.convert("RGB")
        image_arr = np.array(image_rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr)[None,]

        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32, device="cpu")

        i.close()
        return (image_tensor, mask, *star_md)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            mtime = os.path.getmtime(image_path)
            size = os.path.getsize(image_path)
            return f"{mtime}-{size}"
        except Exception:
            return image

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarLoadImagePlus": StarLoadImagePlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarLoadImagePlus": "⭐ Star Load Image+",
}
