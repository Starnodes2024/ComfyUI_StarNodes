import os
import configparser
from typing import Optional, Tuple

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class StarGeminiRefiner:
    def __init__(self):
        self.api_key = self._load_api_key()
        if self.api_key and genai:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"[StarGeminiRefiner] Error configuring API: {e}")

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """Load API key using a pointer ini or a fixed fallback path.
        
        Resolution order:
        1) Node-local ini as pointer: node_dir/googleapi.ini with section [API_PATH] and key 'path' -> points to external ini containing [API_KEY] key.
        2) Node-local ini direct key: node_dir/googleapi.ini with section [API_KEY] key='...'.
        3) Fixed fallback external path: D:\AI\ComfyUINEW\googleapi.ini with section [API_KEY] key='...'.
        """
        node_ini = os.path.join(os.path.dirname(__file__), "googleapi.ini")
        external_default = r"D:\\AI\\ComfyUINEW\\googleapi.ini"

        def _read_key_from_ini(ini_path: str) -> Optional[str]:
            cfg = configparser.ConfigParser()
            try:
                cfg.read(ini_path)
                return cfg.get("API_KEY", "key", fallback=None)
            except Exception:
                return None

        # 1) Pointer ini in node directory
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
                # Direct key mode (backward compatible)
                key_val = cfg.get("API_KEY", "key", fallback=None)
                if key_val:
                    return key_val
            except Exception:
                pass

        # 3) Fixed fallback external path
        if os.path.exists(external_default):
            key_val = _read_key_from_ini(external_default)
            if key_val:
                return key_val

        return None

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "gemini-2.0-flash-thinking-exp-1219",  
            "gemini-2.0-flash-exp",            
        ]
        
        default_system = (
            "You are an expert image prompt engineer. Your goal is to refine the user's input text "
            "into a detailed, high-quality image generation prompt optimized for the google gemini 3 Pro model (nano banana). "
            "Focus on descriptive details, lighting, composition, and style. "
            "Output ONLY the refined prompt, no explanations."
        )

        return {
            "required": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": default_system, "multiline": True}),
                "model": (models, {"default": "gemini-2.0-flash-thinking-exp-1219"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine_prompt"
    CATEGORY = "⭐StarNodes/Text"

    def refine_prompt(self, text_input: str, system_prompt: str, model: str) -> Tuple[str]:
        if not self.api_key:
            return ("Error: API Key not found. Please configure googleapi.ini",)
        
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            # Configure the model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_prompt
            )

            # Generate content
            response = model_instance.generate_content(text_input)
            
            if hasattr(response, 'text'):
                return (response.text,)
            elif hasattr(response, 'parts'):
                # Fallback for some response structures
                return (" ".join([p.text for p in response.parts]),)
            else:
                return ("Error: No response text generated",)

        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "StarGeminiRefiner": StarGeminiRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarGeminiRefiner": "⭐ Star Nano Banana (Gemini Prompter)",
}
