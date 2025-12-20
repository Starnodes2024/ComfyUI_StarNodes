import json
import os
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

from ..text_io.star_save_folder_string import StarSaveFolderString


try:
    from comfy.cli_args import args
except Exception:
    args = None


class StarSaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        preset_options = ["None"] + StarSaveFolderString._load_presets()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "preset_folder": (preset_options, {"default": "None"}),
                "date_folder": ("BOOLEAN", {"default": True}),
                "date_folder_position": (["first", "subfolder"], {"default": "first"}),
                "custom_folder": ("STRING", {"default": "", "multiline": False}),
                "custom_subfolder": ("STRING", {"default": "", "multiline": False}),
                "date_in_filename": (["Off", "prefix", "suffix"], {"default": "Off"}),
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "add_timestamp": ("BOOLEAN", {"default": False}),
                "separator": ("STRING", {"default": "_", "multiline": False}),
            },
            "optional": {
                "StarMetaData 1": ("STRING", {"forceInput": True}),
                "StarMetaData 2": ("STRING", {"forceInput": True}),
                "StarMetaData 3": ("STRING", {"forceInput": True}),
                "StarMetaData 4": ("STRING", {"forceInput": True}),
                "StarMetaData 5": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/IO"
    DESCRIPTION = "Save images like ComfyUI SaveImage, with 5 extra StarMetaData fields stored in PNG metadata."

    def _should_save_default_metadata(self):
        if args is None:
            return True
        return not getattr(args, "disable_metadata", False)

    def _build_star_metadata(self, star_metadata_1, star_metadata_2, star_metadata_3, star_metadata_4, star_metadata_5):
        return {
            "StarMetaData 1": star_metadata_1 or "",
            "StarMetaData 2": star_metadata_2 or "",
            "StarMetaData 3": star_metadata_3 or "",
            "StarMetaData 4": star_metadata_4 or "",
            "StarMetaData 5": star_metadata_5 or "",
        }

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
        prompt=None,
        extra_pnginfo=None,
        **kwargs,
    ):
        filename_prefix, _, _ = StarSaveFolderString().build(
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

        results = []
        save_default_metadata = self._should_save_default_metadata()

        star_metadata_1 = kwargs.get("StarMetaData 1", "")
        star_metadata_2 = kwargs.get("StarMetaData 2", "")
        star_metadata_3 = kwargs.get("StarMetaData 3", "")
        star_metadata_4 = kwargs.get("StarMetaData 4", "")
        star_metadata_5 = kwargs.get("StarMetaData 5", "")

        star_metadata = self._build_star_metadata(
            star_metadata_1, star_metadata_2, star_metadata_3, star_metadata_4, star_metadata_5
        )

        for (batch_number, image) in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            png_info = PngInfo()

            metadata = {}
            if save_default_metadata:
                prompt_info = ""
                if prompt is not None:
                    prompt_info = json.dumps(prompt)
                metadata["prompt"] = prompt_info

                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata[x] = json.dumps(extra_pnginfo[x])

            metadata.update(star_metadata)

            for key, value in metadata.items():
                if not isinstance(value, str):
                    value = str(value)
                png_info.add_text(key, value)

            try:
                for key, value in metadata.items():
                    if not isinstance(value, str):
                        value = str(value)
                    png_info.add(
                        b"comf",
                        key.encode("latin-1", "strict") + b"\0" + value.encode("latin-1", "strict"),
                        after_idat=True,
                    )
            except Exception:
                pass

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=png_info, compress_level=self.compress_level)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "StarSaveImagePlus": StarSaveImagePlus,
    "⭐ Star Save Image+": StarSaveImagePlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSaveImagePlus": "⭐ Star Save Image+",
    "⭐ Star Save Image+": "⭐ Star Save Image+",
}
