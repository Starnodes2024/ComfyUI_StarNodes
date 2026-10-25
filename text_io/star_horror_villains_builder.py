import os
import json
import time
import random

_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json", "star_horror_villains_builder.json")

# Field order used for both the widget layout and the prompt assembly.
_LIST_FIELDS = ["sex", "villain_type", "size", "face", "hair", "skin", "eyes", "body_build",
                "outfit", "weapon", "injuries", "accessories", "aura", "expression", "pose",
                "custom", "style"]

# Face and weapon get dedicated close-up panels on the villain sheet.
_DETAIL_FIELDS = ["face", "weapon"]

_DETERMINERS = ("a ", "an ", "the ", "no ", "his ", "her ", "its ", "one ")


def _natural(text):
    """Add 'a'/'an' in front of natural-language snippets that lack a determiner."""
    low = text.lower()
    if low.startswith(_DETERMINERS):
        return text
    return f"{'an' if low[0] in 'aeiou' else 'a'} {text}"


def _load_data():
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"StarNodes: Could not load {os.path.basename(_JSON_PATH)}: {e}")
        return {}, {}, ""
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    system_prompt = data.get("system_prompt") if isinstance(data.get("system_prompt"), str) else ""
    lists = {k: [str(v) for v in vals] for k, vals in data.items() if isinstance(vals, list)}
    return lists, settings, system_prompt


def _tooltip(field):
    return f"'none' skips this trait, 'random' picks a random entry, 'exclude' keeps it out of the prompt even when random_all is on. Add your own entries in json/star_horror_villains_builder.json (restart ComfyUI to see them in this list, 'random' uses them right away)."


class StarHorrorVillainsBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        lists, _, _ = _load_data()
        required = {}
        for field in _LIST_FIELDS:
            options = ["none", "random", "exclude"] + lists.get(field, [])
            default = "cinematic horror film still" if field == "style" and "cinematic horror film still" in options else "none"
            required[field] = (options, {"default": default, "tooltip": _tooltip(field)})
        required["age"] = ("STRING", {
            "default": "",
            "placeholder": "35 / ageless / random",
            "tooltip": "Free text. Empty = skipped. Type 'random' or enable the random_age toggle for a random age (25-90 by default, change age_random_min/max in the json settings). A plain number becomes 'X year old'."
        })
        required["random_age"] = ("BOOLEAN", {
            "default": False,
            "label_on": "random age (25-90)",
            "label_off": "use age text",
            "tooltip": "If enabled each run, picks a random age between age_random_min and age_random_max from the json settings and ignores the age text field."
        })
        required["additional"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "anything extra, appended at the end",
            "tooltip": "Free text that is appended to the end of the prompt. Empty = skipped."
        })
        required["seed"] = ("INT", {
            "default": 0, "min": 0, "max": 0xffffffffffffffff,
            "tooltip": "Seed for all 'random' picks. 0 = new random picks every run."
        })
        required["random_all"] = ("BOOLEAN", {
            "default": False,
            "label_on": "all random",
            "label_off": "use selections",
            "tooltip": "If enabled, every category gets a random pick regardless of its widget setting. Categories set to 'exclude' stay out of the prompt."
        })
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("description", "villain_sheet", "system_prompt")
    OUTPUT_TOOLTIPS = (
        "The plain generated comma-separated description prompt.",
        "The ready-to-use strict horror villain sheet image prompt for the same villain (multi-panel grid, face/mask and weapon close-up panels). Note: the villain is unclothed unless an outfit is selected or randomized.",
        "The system prompt to feed into your LLM so it can write horror villain sheet prompts itself. Editable as 'system_prompt' in json/star_horror_villains_builder.json.",
    )
    FUNCTION = "build"
    CATEGORY = "⭐StarNodes/Text And Data"

    def _resolve(self, value, options, rng):
        if value in ("none", "exclude"):
            return ""
        if value == "random":
            if not options:
                return ""
            return rng.choice(options)
        return value

    def build(self, sex, villain_type, size, face, hair, skin, eyes, body_build,
              outfit, weapon, injuries, accessories, aura, expression, pose,
              custom, style, age, random_age, additional, seed, random_all):
        lists, settings, system_prompt = _load_data()
        if seed == 0:
            seed = random.randint(0, 0xffffffffffffffff)
        rng = random.Random(seed)

        widget_values = dict(sex=sex, villain_type=villain_type, size=size, face=face, hair=hair, skin=skin,
                             eyes=eyes, body_build=body_build, outfit=outfit, weapon=weapon, injuries=injuries,
                             accessories=accessories, aura=aura, expression=expression, pose=pose,
                             custom=custom, style=style)
        excluded = set(settings.get("exclude", [])) if isinstance(settings.get("exclude"), list) else set()
        excluded |= {f for f in _LIST_FIELDS if widget_values[f] == "exclude"}

        resolved = {}
        for field in _LIST_FIELDS:
            options = lists.get(field, [])
            if random_all and field not in excluded:
                resolved[field] = rng.choice(options) if options else ""
            else:
                resolved[field] = self._resolve(widget_values[field], options, rng)

        age = age.strip()
        if random_age or age.lower() == "random":
            age_min = int(settings.get("age_random_min", 25))
            age_max = int(settings.get("age_random_max", 90))
            age = str(rng.randint(min(age_min, age_max), max(age_min, age_max)))
        if age and age[0].isdigit():
            age = f"{age} year old"

        wear = f"wearing {resolved['outfit']}" if resolved["outfit"] else ""
        armed = f"wielding {resolved['weapon']}" if resolved["weapon"] else ""

        lead = " ".join(x for x in (resolved["size"], age, resolved["sex"], resolved["villain_type"]) if x)

        sheet_prompt = self._build_sheet(resolved, lead, wear, additional)

        parts = []
        if lead:
            parts.append(lead)
        for f in ("skin", "face", "hair", "eyes", "body_build", "injuries"):
            if resolved[f]:
                parts.append(resolved[f])
        if wear:
            parts.append(wear)
        if armed:
            parts.append(armed)
        for f in ("accessories", "aura", "expression", "pose", "custom", "style"):
            if resolved[f]:
                parts.append(resolved[f])
        if additional.strip():
            parts.append(additional.strip())

        return (", ".join(parts), sheet_prompt, system_prompt)

    def _build_sheet(self, resolved, lead, wear, additional):
        lead = lead.replace(" year old", "-year-old") if " year old" in lead else lead
        lead = lead or "adult horror villain"

        traits = [x for x in (resolved["skin"], resolved["hair"], resolved["eyes"],
                              resolved["body_build"], resolved["injuries"], resolved["aura"],
                              resolved["custom"]) if x]
        details = [resolved[f] for f in _DETAIL_FIELDS if resolved[f]]
        items = [self._isolate(resolved["accessories"])] if resolved["accessories"] else []

        sentences = []
        article = "an" if lead[0].lower() in "aeiou" else "a"
        first = f"A horror villain reference sheet of {article} {lead}"
        if resolved["face"]:
            first += f" with {_natural(resolved['face'])}"
        if traits:
            first += f", {', '.join(traits)}"
        if resolved["weapon"]:
            first += f", armed with {resolved['weapon']}"
        first += ", presented in a multi-panel grid layout on a solid, neutral background."
        sentences.append(first)

        sentences.append("The left side has two vertical full-body panels spanning the entire height: one full-body front view and one full-body back view.")

        head = "the top row shows a frontal face portrait, a side profile portrait, and a 45-degree angle portrait"
        face_details = [x for x in (resolved["expression"], _natural(resolved["face"]) if resolved["face"] else "") if x]
        if face_details:
            head += f" showing {' and '.join(face_details)}"
        sentences.append(f"The right side is a 6-panel detail grid in a 3x2 layout: {head}; the bottom row shows the chest area from the front, the pelvic area from the front, and the pelvic area/buttocks from the back.")

        extra = []
        if details:
            closeups = [_natural(resolved["face"]) if resolved["face"] else "", resolved["weapon"]]
            extra.append(f"dedicated close-up panels of {', '.join(c for c in closeups if c)}")
        if items:
            extra.append(f"separate isolated close-ups of the following items on neutral backgrounds, never worn or held by the villain: {', '.join(items)}")
        if extra:
            sentences.append(f"The right-side grid expands to 8 panels in a 4x2 layout, adding {' and '.join(extra)}.")

        if wear:
            sentences.append(f"The villain wears {wear.removeprefix('wearing ')} in the full-body panels.")
        else:
            sentences.append("The villain is unclothed in the full-body panels.")

        style = resolved["style"]
        style_sentence = "The style is clinical and scientific, like a forensic case file"
        if style:
            style_sentence = f"Rendered in {style} style, presented in a clinical and scientific way like a forensic case file"
        style_sentence += ", with flat, even studio lighting that clearly illuminates all details without harsh shadows, and enforces absolute consistency in the villain's facial features, proportions, and overall appearance across every panel."
        sentences.append(style_sentence)
        sentences.append("No text, letters, numbers, symbols, watermarks, charts, or labels appear anywhere in the image.")

        prompt = " ".join(sentences)
        if additional.strip():
            prompt += ", " + additional.strip()
        return prompt

    def _isolate(self, text):
        for pre in ("holding a ", "holding ", "wearing a ", "wearing ", "with a ", "with "):
            if text.lower().startswith(pre):
                return text[len(pre):]
        return text

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "StarHorrorVillainsBuilder": StarHorrorVillainsBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarHorrorVillainsBuilder": "⭐ Star Horror Villains Builder",
}
