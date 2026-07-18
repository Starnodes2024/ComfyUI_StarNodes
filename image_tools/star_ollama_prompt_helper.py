import os
import json
import base64
import re
from io import BytesIO

import numpy as np
from PIL import Image

try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[StarOllamaPromptHelper] Warning: 'ollama' not installed. Node will not be available.")
    Client = None

try:
    from server import PromptServer
    from aiohttp import web
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False
    print("[StarOllamaPromptHelper] Warning: ComfyUI server not available.")

_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "systemprompts.json")

# Downscale images sent to vision models so prompt processing is faster.
# Set to None to disable and always send full resolution.
_MAX_IMAGE_SIZE = 768


def _load_presets():
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(sorted(data.items()))
    except Exception:
        return {}


_PRESETS = _load_presets()
_PRESET_NAMES = list(_PRESETS.keys())


if SERVER_AVAILABLE and OLLAMA_AVAILABLE:
    @PromptServer.instance.routes.post("/starnodes/ollama/models")
    async def _star_ollama_models(request):
        body = await request.json()
        url = body.get("url", "http://127.0.0.1:11434")
        try:
            client = Client(host=url)
            raw = client.list().get("models", [])
            names = [m.get("model", m.get("name", "")) for m in raw]
            return web.json_response(names)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


class StarOllamaPromptHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_address": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "multiline": False,
                    "tooltip": "Ollama server address.",
                }),
                "model": ((), {
                    "tooltip": "Select a model. Click refresh to load available models from the server.",
                }),
                "keep_alive": (["0", "5m", "30m", "1h", "-1"], {
                    "default": "5m",
                    "tooltip": "How long the model stays loaded in memory after generation. '-1' = forever, '0' = unload immediately (slow on repeat runs).",
                }),
                "system_prompt_preset": (["Custom"] + _PRESET_NAMES, {
                    "default": "Custom",
                    "tooltip": "Choose a preset system prompt or 'Custom' to write your own.",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom system prompt (used when preset is 'Custom').",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your prompt text to send to the model.",
                }),
                "allow_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, outputs the reasoning process. If False, instructs the model to skip reasoning to speed up generation.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Higher values produce more creative output.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2 ** 31 - 1,
                    "control_after_generate": True,
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision language models. Refer to it in your prompt as 'this image'.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "think_output")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Prompts"
    DESCRIPTION = "Use a local Ollama model to create or refine prompts for image generation."

    def generate(self, local_address, model, keep_alive, system_prompt_preset,
                 system_prompt, prompt, allow_thinking, temperature, seed, image=None):

        if system_prompt_preset != "Custom" and system_prompt_preset in _PRESETS:
            sys_text = _PRESETS[system_prompt_preset]
        else:
            sys_text = system_prompt

        # Optimization: Force the model to skip thinking if disabled
        if not allow_thinking:
            sys_text += "\n\nImportant: Answer directly without any internal reasoning or <think> tags."

        opts = {"temperature": float(temperature)}
        if seed > 0:
            opts["seed"] = int(seed)

        imgs_b64 = []
        if image is not None:
            for img_tensor in image:
                i = 255.0 * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                if _MAX_IMAGE_SIZE and max(img.size) > _MAX_IMAGE_SIZE:
                    img.thumbnail((_MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE), Image.BILINEAR)
                buf = BytesIO()
                img.save(buf, format="PNG", optimize=False)
                imgs_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        client = Client(host=local_address)
        
        gen_kwargs = {
            "model": model,
            "prompt": prompt,
            "system": sys_text,
            "options": opts,
            "keep_alive": keep_alive,
            "think": allow_thinking
        }
        
        if imgs_b64:
            gen_kwargs["images"] = imgs_b64
            
        resp = client.generate(**gen_kwargs)

        if isinstance(resp, dict):
            result_text = resp.get("response", "")
            think_text = resp.get("thinking", "")
        else:
            result_text = getattr(resp, "response", "")
            think_text = getattr(resp, "thinking", "")
            
        # Optionaler Fallback für Modelle, die das native Thinking-Feld ignorieren
        if not think_text:
            think_match = re.search(r"<think>(.*?)</think>", result_text, re.DOTALL | re.IGNORECASE)
            if think_match:
                think_text = think_match.group(1).strip()
                result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL | re.IGNORECASE).strip()

        if think_text is None:
            think_text = ""

        # Erzwinge leeren Output, wenn Toggle auf False steht
        if not allow_thinking:
            think_text = ""

        return {"result": (result_text, think_text), "ui": {"seed": [seed]}}


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if OLLAMA_AVAILABLE and SERVER_AVAILABLE:
    NODE_CLASS_MAPPINGS["StarOllamaPromptHelper"] = StarOllamaPromptHelper
    NODE_DISPLAY_NAME_MAPPINGS["StarOllamaPromptHelper"] = "⭐ Star Ollama Prompt Helper"
else:
    print("[StarOllamaPromptHelper] Node not registered due to missing dependencies: ollama")