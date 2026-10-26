import os
import json
import time
import random

_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json", "star_sexy_builder.json")

# Field order used for both the widget layout and the prompt assembly.
_LIST_FIELDS = ["sex", "hairstyle", "hair_color", "skin_tone", "body_build", "breasts", "chest",
                "nipples", "waist", "butt", "legs", "genitals", "pubic_hair",
                "lingerie", "nightwear", "kinky_outfit", "accessories", "expression", "pose",
                "ambiance", "custom", "style"]

# Wear categories are mutually exclusive under random_all - only one of them is picked.
_WEAR_FIELDS = ["lingerie", "nightwear", "kinky_outfit"]

# Anatomy categories that get dedicated close-up panels on the figure sheet.
_DETAIL_FIELDS = ["breasts", "chest", "butt"]


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
    return f"'none' skips this trait, 'random' picks a random entry, 'exclude' keeps it out of the prompt even when random_all is on. Add your own entries in json/star_sexy_builder.json (restart ComfyUI to see them in this list, 'random' uses them right away)."


class StarSexyBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        lists, _, _ = _load_data()
        required = {}
        for field in _LIST_FIELDS:
            options = ["none", "random", "exclude"] + lists.get(field, [])
            default = "boudoir photography" if field == "style" and "boudoir photography" in options else "none"
            required[field] = (options, {"default": default, "tooltip": _tooltip(field)})
        required["age"] = ("STRING", {
            "default": "",
            "placeholder": "25 / young adult / random",
            "tooltip": "Free text. Empty = skipped. Type 'random' or enable the random_age toggle for a random age (18-60 by default, change age_random_min/max in the json settings). A plain number becomes 'X year old'."
        })
        required["random_age"] = ("BOOLEAN", {
            "default": False,
            "label_on": "random age (18-60)",
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
            "tooltip": "If enabled, every category gets a random pick regardless of its widget setting. Only one of lingerie/nightwear/kinky_outfit is picked. Categories set to 'exclude' stay out of the prompt."
        })
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("description", "sexy_sheet", "system_prompt")
    OUTPUT_TOOLTIPS = (
        "The plain generated comma-separated description prompt.",
        "The ready-to-use strict figure sheet image prompt for the same character (multi-panel grid, anatomy close-up panels). Note: the character is unclothed unless a wear category is selected or randomized.",
        "The system prompt to feed into your LLM so it can write figure sheet prompts itself. Editable as 'system_prompt' in json/star_sexy_builder.json.",
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

    def build(self, sex, hairstyle, hair_color, skin_tone, body_build, breasts, chest, nipples,
              waist, butt, legs, genitals, pubic_hair, lingerie, nightwear, kinky_outfit,
              accessories, expression, pose, ambiance, custom, style,
              age, random_age, additional, seed, random_all):
        lists, settings, system_prompt = _load_data()
        if seed == 0:
            seed = random.randint(0, 0xffffffffffffffff)
        rng = random.Random(seed)

        widget_values = dict(sex=sex, hairstyle=hairstyle, hair_color=hair_color, skin_tone=skin_tone,
                             body_build=body_build, breasts=breasts, chest=chest,
                             nipples=nipples, waist=waist, butt=butt, legs=legs, genitals=genitals,
                             pubic_hair=pubic_hair, lingerie=lingerie, nightwear=nightwear, kinky_outfit=kinky_outfit,
                             accessories=accessories, expression=expression, pose=pose, ambiance=ambiance,
                             custom=custom, style=style)
        excluded = set(settings.get("exclude", [])) if isinstance(settings.get("exclude"), list) else set()
        excluded |= {f for f in _LIST_FIELDS if widget_values[f] == "exclude"}
        wear_pick = ""
        if random_all:
            pool = [f for f in _WEAR_FIELDS if f not in excluded and lists.get(f)]
            if pool:
                wear_pick = rng.choice(pool)

        resolved = {}
        for field in _LIST_FIELDS:
            options = lists.get(field, [])
            if random_all and field not in excluded:
                if field in _WEAR_FIELDS:
                    resolved[field] = rng.choice(options) if field == wear_pick and options else ""
                else:
                    resolved[field] = rng.choice(options) if options else ""
            else:
                resolved[field] = self._resolve(widget_values[field], options, rng)

        age = age.strip()
        if random_age or age.lower() == "random":
            age_min = int(settings.get("age_random_min", 18))
            age_max = int(settings.get("age_random_max", 60))
            age = str(rng.randint(min(age_min, age_max), max(age_min, age_max)))
        if age and age[0].isdigit():
            age = f"{age} year old"

        outfits = [resolved[f] for f in _WEAR_FIELDS]
        outfits = [o for o in outfits if o]
        wear = f"wearing {' and '.join(outfits)}" if outfits else ""

        hair = ""
        hair_style, color = resolved["hairstyle"], resolved["hair_color"]
        if hair_style and color:
            hair = hair_style.replace("hair", f"{color} hair", 1) if "hair" in hair_style else f"{color} {hair_style}"
        elif hair_style:
            hair = hair_style
        elif color:
            hair = f"{color} hair"

        lead = " ".join(x for x in (age, resolved["sex"]) if x)

        sheet_prompt = self._build_sheet(resolved, lead, hair, wear, additional)

        parts = []
        if lead:
            parts.append(lead)
        if hair:
            parts.append(hair)
        for f in ("skin_tone", "body_build", "breasts", "chest", "nipples", "waist", "butt", "legs",
                  "genitals", "pubic_hair"):
            if resolved[f]:
                parts.append(resolved[f])
        if wear:
            parts.append(wear)
        for f in ("accessories", "expression", "pose", "ambiance", "custom", "style"):
            if resolved[f]:
                parts.append(resolved[f])
        if additional.strip():
            parts.append(additional.strip())

        return (", ".join(parts), sheet_prompt, system_prompt)

    def _build_sheet(self, resolved, lead, hair, wear, additional):
        lead = lead.replace(" year old", "-year-old") if " year old" in lead else lead
        lead = lead or "adult person"

        traits = [x for x in (hair, resolved["skin_tone"], resolved["body_build"], resolved["waist"],
                              resolved["legs"], resolved["custom"]) if x]
        closeups = [f"the {resolved[f]}" for f in _DETAIL_FIELDS if resolved[f]]
        items = [self._isolate(resolved["accessories"])] if resolved["accessories"] else []

        sentences = []
        article = "an" if lead[0].lower() in "aeiou" else "a"
        first = f"An adult figure reference sheet of {article} {lead}"
        if traits:
            first += f" with {', '.join(traits)}"
        anatomy = [x for x in (resolved["breasts"], resolved["chest"], resolved["nipples"],
                               resolved["butt"]) if x]
        if anatomy:
            first += f", featuring {', '.join(anatomy)}"
        first += ", presented in a multi-panel grid layout on a solid, neutral background."
        sentences.append(first)

        sentences.append("The left side has two vertical full-body panels spanning the entire height: one full-body front view and one full-body back view.")

        head = "the top row shows a frontal face portrait, a side profile portrait, and a 45-degree angle portrait"
        if resolved["expression"]:
            head += f" with {resolved['expression']}"
        chest_row = "the chest area from the front"
        chest_detail = ", ".join(x for x in (resolved["breasts"], resolved["chest"], resolved["nipples"]) if x)
        if chest_detail:
            chest_row += f" (showing {chest_detail})"
        pelvic_row = "the pelvic area from the front"
        pelvic_detail = ", ".join(x for x in (resolved["genitals"], resolved["pubic_hair"]) if x)
        if pelvic_detail:
            pelvic_row += f" (clearly showing {pelvic_detail})"
        butt_row = "the buttocks from the back"
        if resolved["butt"]:
            butt_row += f" (showing the {resolved['butt']})"
        sentences.append(f"The right side is a 6-panel detail grid in a 3x2 layout: {head}; the bottom row shows {chest_row}, {pelvic_row}, and {butt_row}.")

        extra = []
        if closeups:
            extra.append(f"dedicated close-up panels of {', '.join(closeups)}")
        if items:
            extra.append(f"separate isolated close-ups of the following items on neutral backgrounds, never worn or held by the subject: {', '.join(items)}")
        if extra:
            sentences.append(f"The right-side grid expands to 8 panels in a 4x2 layout, adding {' and '.join(extra)}.")

        if wear:
            sentences.append(f"The subject wears {wear.removeprefix('wearing ')} in the full-body panels.")
        else:
            sentences.append("The subject is unclothed in the full-body panels.")

        style = resolved["style"]
        style_sentence = "The style is clinical and objective, like a figure study plate"
        if style:
            style_sentence = f"Rendered in {style} style, presented in a clinical and objective way like a figure study plate"
        style_sentence += ", with flat, even studio lighting that clearly illuminates all anatomical details without harsh shadows, and enforces absolute consistency in the subject's facial features, proportions, and overall appearance across every panel."
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
    "StarSexyBuilder": StarSexyBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSexyBuilder": "⭐ Star Sexy Builder",
}
