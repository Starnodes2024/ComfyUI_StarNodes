import os
import json
import random

import torch

import folder_paths
import comfy.sd

# Shared with the Star Easy-Text-Storage and Star Ollama Prompt Helper nodes:
# all read the same JSON so entries are immediately available as presets here.
_STORAGE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "json", "startext.json")

_DEFAULT_MODEL = "qwen3-vl-4b-heretic_int8.safetensors"


def _load_presets():
    try:
        with open(_STORAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(sorted(data.items()))
    except Exception:
        return {}


class StarClipPromptRefiner:
    """
    Load a text encoder (e.g. a Qwen3-VL instruct model) from models/text_encoders
    and use it to generate or refine prompts, optionally guided by reference images.
    """
    CATEGORY = "⭐StarNodes/Prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "think_output")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = "Refine or generate prompts with a local text encoder (CLIP) that supports text generation, e.g. Qwen3-VL. Supports up to 4 reference images and shared system prompt presets."

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("text_encoders")
        default_model = _DEFAULT_MODEL if _DEFAULT_MODEL in models else (models[0] if models else _DEFAULT_MODEL)
        return {
            "required": {
                "clip_name": (models, {
                    "default": default_model,
                    "tooltip": "Text encoder from models/text_encoders used as the language model. Vision-language models like Qwen3-VL can also read the reference images.",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your prompt text to send to the model.",
                }),
                "system_prompt_preset": (["Custom"] + list(_load_presets().keys()), {
                    "default": "Custom",
                    "tooltip": "Choose a preset system prompt or 'Custom' to write your own. Presets are loaded from the shared text storage (json/startext.json).",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom system prompt (used when preset is 'Custom'). Replaces the system prompt in the model's chat template.",
                }),
                "use_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: the seed below is used and the node caches its result when nothing changes. Off: a random seed is used every run, so the node always executes again even if the inputs did not change.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2 ** 31 - 1,
                    "control_after_generate": True,
                    "tooltip": "Seed for reproducible output sampling (only used when 'use_seed' is on).",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Higher values produce more creative output, lower values more focused/deterministic.",
                }),
                "top_k": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Limits sampling to the top-K most likely tokens. 0 disables the limit.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling: only tokens within this cumulative probability are considered.",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum token probability relative to the most likely token. 0 disables.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Penalizes repeated tokens. 1.0 disables the penalty.",
                }),
                "max_length": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 32768,
                    "tooltip": "Maximum number of tokens to generate.",
                }),
                "thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Operate in thinking mode if the model supports it. The reasoning is returned on think_output. Note: with some Qwen3-VL models (e.g. heretic) turning this off injects an empty think block that can confuse the model - keep it on if output looks wrong.",
                }),
            },
            "optional": {
                "clip_override": ("CLIP", {
                    "tooltip": "Optional: use an already loaded text encoder (CLIP) from the workflow instead of loading the model from the dropdown. The clip_name dropdown is ignored while this is connected.",
                }),
                "image1": ("IMAGE", {
                    "tooltip": "Optional reference image for vision language models. Connecting an image adds another slot (up to 4).",
                }),
                "video": ("IMAGE", {
                    "tooltip": "Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally. Only used by models that support video (e.g. Gemma 4).",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio input, e.g. from a Load Audio node. Only used by models that support audio (e.g. Gemma 4).",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, use_seed, **kwargs):
        if use_seed:
            return ""  # cache like any other node when the seed is fixed
        return float("nan")  # always run again

    def generate(self, clip_name, prompt, system_prompt_preset, system_prompt,
                 use_seed, seed, temperature, top_k, top_p, min_p,
                 repetition_penalty, max_length, thinking, **kwargs):

        presets = _load_presets()
        if system_prompt_preset != "Custom" and system_prompt_preset in presets:
            sys_text = presets[system_prompt_preset]
        else:
            sys_text = system_prompt

        # Collect reference images (dynamic image1..image4 slots) and split batches into single frames
        frames = []
        for i in range(1, 5):
            img = kwargs.get(f"image{i}")
            if img is None:
                continue
            for j in range(img.shape[0]):
                frames.append(img[j:j + 1])

        media_kwargs = {}
        if frames:
            if all(f.shape[1] == frames[0].shape[1] and f.shape[2] == frames[0].shape[2] for f in frames):
                media_kwargs["image"] = torch.cat(frames, dim=0)
            else:
                # Different resolutions cannot be batched; pass individually (supported by Qwen3-VL)
                media_kwargs["images"] = frames

        video = kwargs.get("video")
        if video is not None:
            media_kwargs["video"] = video
        audio = kwargs.get("audio")
        if audio is not None:
            media_kwargs["audio"] = audio

        clip = kwargs.get("clip_override")
        if clip is None:
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"))

        run_seed = int(seed) if use_seed else random.randint(0, 2 ** 31 - 1)

        tokens = clip.tokenize(prompt, min_length=1, thinking=thinking, system_prompt=sys_text, **media_kwargs)

        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=run_seed,
        )

        generated_text = clip.decode(generated_ids)

        # Reasoning opens in the generated text or at the end of a raw prompt, and has no close when max_length cut it short.
        reasoning, _, text = generated_text.partition("</think>")
        if not reasoning.lstrip().startswith("<think>") and not prompt.rstrip().endswith("<think>"):
            reasoning, text = "", generated_text
        if not thinking:
            reasoning = ""

        return {"result": (text.strip(), reasoning.replace("<think>", "", 1).strip()), "ui": {"seed": [run_seed]}}


NODE_CLASS_MAPPINGS = {
    "StarClipPromptRefiner": StarClipPromptRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarClipPromptRefiner": "⭐ Star CLIP Prompt Refiner",
}
