import os
import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from ollama import Client
from server import PromptServer
from aiohttp import web

_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "systemprompts.json")


def _load_presets():
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(sorted(data.items()))
    except Exception:
        return {}


_PRESETS = _load_presets()
_PRESET_NAMES = list(_PRESETS.keys())


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
                "free_ram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: unload model immediately after generation. Off: keep model loaded for 5 minutes.",
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Prompts"
    DESCRIPTION = "Use a local Ollama model to create or refine prompts for image generation."

    def generate(self, local_address, model, free_ram, system_prompt_preset,
                 system_prompt, prompt, temperature, seed, image=None):

        if system_prompt_preset != "Custom" and system_prompt_preset in _PRESETS:
            sys_text = _PRESETS[system_prompt_preset]
        else:
            sys_text = system_prompt

        keep_alive = "0" if free_ram else "5m"

        opts = {"temperature": float(temperature)}
        if seed > 0:
            opts["seed"] = int(seed)

        user_msg = {"role": "user", "content": prompt}
        if image is not None:
            imgs_b64 = []
            for img_tensor in image:
                i = 255.0 * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buf = BytesIO()
                img.save(buf, format="PNG")
                imgs_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            user_msg["images"] = imgs_b64

        client = Client(host=local_address)
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": sys_text},
                user_msg,
            ],
            options=opts,
            keep_alive=keep_alive,
        )

        result_text = resp.message.content
        return {"result": (result_text,), "ui": {"seed": [seed]}}


NODE_CLASS_MAPPINGS = {
    "StarOllamaPromptHelper": StarOllamaPromptHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarOllamaPromptHelper": "⭐ Star Ollama Prompt Helper",
}
