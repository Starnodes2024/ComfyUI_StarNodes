import os
import json
from typing import Dict, Any, Tuple


class StarImageEditQwenKontext:
    PROMPTS_FILENAME = "editprompts.json"

    @classmethod
    def _prompts_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), cls.PROMPTS_FILENAME)

    def _load_prompts(self) -> Dict[str, Any]:
        path = self._prompts_path()
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            return {}
        if not hasattr(self, "_cache"):
            self._cache = {"mtime": None, "data": {}}
        if self._cache.get("mtime") != mtime:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = {"mtime": mtime, "data": data}
            except Exception:
                # keep previous data if any
                pass
        return self._cache.get("data", {})

    @classmethod
    def INPUT_TYPES(cls):
        # These UI choices will be hydrated at runtime; provide fallbacks for graph save-load
        model_choices = [
            "Qwen Image Edit",
            "FLUX.1 Kontext",
        ]
        task_choices = [
            "Change Color",
            "Replace Background",
            "Add Object",
            "Remove Object",
            "Style Transfer",
            "Edit Text In Image",
            "Change Lighting",
            "Change Expression",
            "Change Clothing",
            "Image Restore",
        ]
        return {
            "required": {
                "model": (model_choices, {"default": "Qwen Image Edit"}),
                "task": (task_choices, {"default": "Change Color"}),
            },
            "optional": {
                # Common parameter fields used by templates; user can leave unused ones empty
                "subject": ("STRING", {"default": "", "multiline": False}),
                "color": ("STRING", {"default": "", "multiline": False}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "object": ("STRING", {"default": "", "multiline": False}),
                "location": ("STRING", {"default": "", "multiline": False}),
                "style": ("STRING", {"default": "", "multiline": False}),
                "surface": ("STRING", {"default": "", "multiline": False}),
                "text": ("STRING", {"default": "", "multiline": False}),
                "lighting": ("STRING", {"default": "", "multiline": False}),
                "expression": ("STRING", {"default": "", "multiline": False}),
                "clothing_item": ("STRING", {"default": "", "multiline": False}),
                "style_or_color": ("STRING", {"default": "", "multiline": False}),
                # Advanced controls
                "keep_clause": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "params_hint")
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/Prompts"

    def _available_for_model(self, model_name: str) -> Tuple[list, dict]:
        data = self._load_prompts() or {}
        models = (data.get("models") or {})
        m = models.get(model_name) or {}
        tasks = m.get("tasks") or {}
        return list(tasks.keys()), tasks

    def _interpolate(self, template: str, params: Dict[str, Any]) -> str:
        # Simple {param} replacement; leave braces if missing so the user sees what's needed
        out = template
        for k, v in params.items():
            if v is None:
                continue
            try:
                out = out.replace("{" + k + "}", str(v))
            except Exception:
                pass
        return out

    def build(
        self,
        model: str,
        task: str,
        subject: str = "",
        color: str = "",
        background: str = "",
        object: str = "",
        location: str = "",
        style: str = "",
        surface: str = "",
        text: str = "",
        lighting: str = "",
        expression: str = "",
        clothing_item: str = "",
        style_or_color: str = "",
        keep_clause: str = "",
    ):
        # Resolve tasks for model
        task_list, tasks_map = self._available_for_model(model)
        tdef = tasks_map.get(task) or {}
        if not tdef:
            available_tasks = ", ".join([f"'{t}'" for t in task_list]) if task_list else "none"
            return (f"Error: Task '{task}' is not available for model '{model}'. Available tasks: {available_tasks}", f"Available tasks: {available_tasks}")
        template = tdef.get("template") or ""
        # Merge params
        params: Dict[str, Any] = {
            "subject": subject.strip(),
            "color": color.strip(),
            "background": background.strip(),
            "object": object.strip(),
            "location": location.strip(),
            "style": style.strip(),
            "surface": surface.strip(),
            "text": text.strip(),
            "lighting": lighting.strip(),
            "expression": expression.strip(),
            "clothing_item": clothing_item.strip(),
            "style_or_color": style_or_color.strip(),
        }
        # Interpolate
        prompt = self._interpolate(template, params)
        # Keep clause injection at the end if provided
        keep_clause = (keep_clause or "").strip()
        if keep_clause:
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = prompt + f" {keep_clause}"
        # Fallback if template empty
        if not prompt.strip():
            prompt = f"[{model}] {task}. Fill parameters in the node or edit editprompts.json."
        # Build params hint string for UI
        required_list = tdef.get("params") or []
        def _is_filled(k: str) -> bool:
            v = params.get(k)
            return bool(v is not None and str(v).strip() != "")
        missing = [k for k in required_list if not _is_filled(k)]
        if required_list:
            hint = "needed: " + ", ".join(required_list)
            if missing:
                hint += " | missing: " + ", ".join(missing)
        else:
            hint = "no specific placeholders required"
        return (prompt, hint)


NODE_CLASS_MAPPINGS = {
    "StarImageEditQwenKontext": StarImageEditQwenKontext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageEditQwenKontext": "⭐ Star Image Edit for Qwen/Kontext",
}
