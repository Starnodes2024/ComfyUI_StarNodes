import os
from datetime import datetime
import json


class StarSaveFolderString:
    PRESETS_FILENAME = "star_save_folder_presets.json"

    @classmethod
    def _presets_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), cls.PRESETS_FILENAME)

    @classmethod
    def _load_presets(cls):
        defaults = [
            "Images",
            "Video",
            "Inpaint",
            "Controlnet",
            "Edits",
            "Faceswap",
            "Creatures",
            "Backdrops",
        ]
        try:
            with open(cls._presets_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                cleaned = []
                for x in data:
                    if isinstance(x, str):
                        s = x.strip()
                        if s and s not in cleaned:
                            cleaned.append(s)
                return cleaned or defaults
        except Exception:
            pass
        return defaults

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = ["None"] + cls._load_presets()
        return {
            "required": {
                "preset_folder": (preset_options, {"default": "None"}),
                "date_folder": ("BOOLEAN", {"default": True}),
                "date_folder_position": (["first", "subfolder"], {"default": "first"}),
                "custom_folder": ("STRING", {"default": "ComfyUi", "multiline": False}),
                "custom_subfolder": ("STRING", {"default": "", "multiline": False}),
                "date_in_filename": (["Off", "prefix", "suffix"], {"default": "Off"}),
                "filename": ("STRING", {"default": "Image", "multiline": False}),
                "add_timestamp": ("BOOLEAN", {"default": False}),
                "separator": ("STRING", {"default": "_", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/IO"

    def _safe_part(self, s: str) -> str:
        # Remove characters that are generally invalid in file/dir names across OSes
        invalid = '<>:"/\\|?*\n\r\t'
        return "".join(c for c in s if c not in invalid).strip()

    def build(
        self,
        preset_folder: str = "None",
        date_folder: bool = True,
        date_folder_position: str = "first",
        custom_folder: str = "ComfyUi",
        custom_subfolder: str = "",
        date_in_filename: str = "Off",
        filename: str = "Image",
        add_timestamp: bool = False,
        separator: str = "_",
    ):
        # Normalize and sanitize inputs
        preset_folder = (preset_folder or "None").strip()
        custom_folder = self._safe_part(custom_folder or "")
        custom_subfolder = self._safe_part(custom_subfolder or "")
        filename = self._safe_part(filename or "Image") or "Image"
        separator = self._safe_part(separator or "_") or "_"

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        # Base folder stack (without date)
        base_parts = []
        if preset_folder and preset_folder != "None":
            base_parts.append(self._safe_part(preset_folder))
        if custom_folder:
            base_parts.append(custom_folder)
        if custom_subfolder:
            base_parts.append(custom_subfolder)

        # Inject date folder per setting
        if date_folder:
            if (date_folder_position or "first").lower() == "first":
                parts = [date_str] + base_parts
            else:
                parts = base_parts + [date_str]
        else:
            parts = base_parts

        # Build filename with optional date and time
        name = filename
        if date_in_filename == "prefix":
            name = f"{date_str}{filename}"
        elif date_in_filename == "suffix":
            name = f"{filename}{date_str}"

        if add_timestamp:
            # custom separator before timestamp
            name = f"{name}{separator}{time_str}"

        # Join using posix-like forward slashes for ComfyUI consistency across OS
        folder_path = "/".join(p for p in parts if p)
        path = f"{folder_path}/{name}" if folder_path else name

        return (path,)


NODE_CLASS_MAPPINGS = {
    "StarSaveFolderString": StarSaveFolderString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSaveFolderString": "⭐ Star Save Folder String",
}
