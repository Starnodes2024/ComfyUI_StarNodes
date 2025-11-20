import json
import os
from typing import List, Dict, Any, Tuple


class StarOllamaSysprompterJC:
    """
    Build two prompt strings for Ollama-style prompting:
    - System Prompt (STRING)
    - Detail Prompt (STRING)

    Inputs:
    - max_tokens (INT): Maximum tokens to aim for in detail prompt. Default 300.
    - style (CHOICE): Style dropdown loaded from styles.json. First option is "Own Style".
    - own_style (STRING): Used as both STYLENAME and STYLE when style == "Own Style".
    - additional_system_prompt (STRING): Appended at the end of the system prompt when provided.

    The styles.json file should be an array of objects: [{"name": "Pencil Sketch", "style": "detailed pencil sketch"}, ...]
    Located alongside this node file.
    """

    _STYLES_CACHE: List[Dict[str, str]] | None = None

    @classmethod
    def _styles_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "styles.json")

    @classmethod
    def _load_styles(cls) -> List[Dict[str, str]]:
        if cls._STYLES_CACHE is not None:
            return cls._STYLES_CACHE
        path = cls._styles_path()
        styles: List[Dict[str, str]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item and "style" in item:
                            styles.append({"name": str(item["name"]), "style": str(item["style"])})
        except Exception:
            # Fallback to minimal built-in styles if file missing or invalid
            styles = [
                {"name": "Pencil Sketch", "style": "detailed pencil sketch"},
                {"name": "Photorealistic", "style": "highly detailed photorealistic"},
            ]
        cls._STYLES_CACHE = styles
        return styles

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        styles = cls._load_styles()
        options = ["Own Style"] + [s["name"] for s in styles]
        return {
            "required": {
                "max_tokens": ("INT", {"default": 300, "min": 1, "max": 8192, "step": 1}),
                "style": (options, {"default": options[0]}),
                "own_style": ("STRING", {"default": "", "multiline": False}),
                "additional_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "fit_composition_to_style": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "detail_prompt", "style_name")
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/Prompts"

    @classmethod
    def _resolve_style(cls, style_choice: str, own_style: str) -> Tuple[str, str]:
        """Return (style_name, style_desc)"""
        if style_choice == "Own Style":
            text = own_style.strip() or "Custom Style"
            return text, text
        # find in loaded styles
        for s in cls._load_styles():
            if s["name"] == style_choice:
                return s["name"], s["style"]
        # fallback
        return style_choice, style_choice

    def build(self, max_tokens: int, style: str, own_style: str, additional_system_prompt: str, fit_composition_to_style: bool) -> Tuple[str, str, str]:
        style_name, style_desc = self._resolve_style(style, own_style)

        # System prompt
        sys_parts = [
            "You are an AI artist. you create one image prompt. no questions or comments.",
        ]
        add = (additional_system_prompt or "").strip()
        if add:
            sys_parts.append(add)
        if bool(fit_composition_to_style):
            sys_parts.append("Change image composition to fit the chosen style.")
        system_prompt = " ".join(sys_parts)

        # Detail prompt
        # Replace ### with max_tokens inside the sentence as specified
        detail_prompt = (
            f"describe the image and create an image prompt with {max_tokens} tokens max.  "
            f"no questions or comments. change style to {style_name}. turn prompt into {style_desc}"
        )

        return (system_prompt, detail_prompt, style_name)


NODE_CLASS_MAPPINGS = {
    "StarOllamaSysprompterJC": StarOllamaSysprompterJC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarOllamaSysprompterJC": "⭐ Star Ollama Sysprompter (JC)",
}
