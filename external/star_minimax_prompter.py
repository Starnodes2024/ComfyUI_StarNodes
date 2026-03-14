import os
import configparser
from typing import Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class StarMiniMaxPrompter:
    def __init__(self):
        self.api_key = self._load_api_key()

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """Load MiniMax API key from ini file or environment variable.

        Resolution order:
        1) Root-level ini as pointer: comfyui_starnodes/minimaxapi.ini with section [API_PATH] and key 'path' -> points to external ini containing [API_KEY] key.
        2) Root-level ini direct key: comfyui_starnodes/minimaxapi.ini with section [API_KEY] key='...'.
        3) Environment variable: MINIMAX_API_KEY
        """
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        node_ini = os.path.join(root_dir, "minimaxapi.ini")

        def _read_key_from_ini(ini_path: str) -> Optional[str]:
            cfg = configparser.ConfigParser()
            try:
                cfg.read(ini_path)
                return cfg.get("API_KEY", "key", fallback=None)
            except Exception:
                return None

        # 1) Pointer ini in root directory
        if os.path.exists(node_ini):
            cfg = configparser.ConfigParser()
            try:
                cfg.read(node_ini)
                # Pointer mode
                if cfg.has_section("API_PATH"):
                    pointer_path = cfg.get("API_PATH", "path", fallback=None)
                    if pointer_path:
                        pointer_path = os.path.expandvars(pointer_path)
                        if os.path.exists(pointer_path):
                            key_val = _read_key_from_ini(pointer_path)
                            if key_val:
                                return key_val
                # Direct key mode
                key_val = cfg.get("API_KEY", "key", fallback=None)
                if key_val:
                    return key_val
            except Exception:
                pass

        # 3) Environment variable fallback
        env_key = os.environ.get("MINIMAX_API_KEY")
        if env_key:
            return env_key

        return None

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ]

        default_system = (
            "You are an expert image prompt engineer. Your goal is to refine the user's input text "
            "into a detailed, high-quality image generation prompt. "
            "Focus on descriptive details, lighting, composition, and style. "
            "Output ONLY the refined prompt, no explanations."
        )

        return {
            "required": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": default_system, "multiline": True}),
                "model": (models, {"default": "MiniMax-M2.5"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 204800, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine_prompt"
    CATEGORY = "⭐StarNodes/Text"

    def refine_prompt(
        self,
        text_input: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str]:
        if not self.api_key:
            return ("Error: MiniMax API Key not found. Please configure minimaxapi.ini or set MINIMAX_API_KEY environment variable.",)

        if OpenAI is None:
            return ("Error: openai package not installed. Install with: pip install openai",)

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.minimax.io/v1",
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response.choices and response.choices[0].message.content:
                return (response.choices[0].message.content,)
            else:
                return ("Error: No response text generated",)

        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "StarMiniMaxPrompter": StarMiniMaxPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMiniMaxPrompter": "⭐ Star MiniMax Prompter",
}
