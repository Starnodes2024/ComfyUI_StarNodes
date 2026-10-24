import os
import json
import time
import random

_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json", "star_creature_builder.json")

# Field order used for both the widget layout and the prompt assembly.
_LIST_FIELDS = ["sex", "creature_type", "element", "body_plan", "size", "head", "ears", "horns", "mouth",
                "skin_type", "body_color", "pattern", "pattern_color", "hair_mane", "hair_color",
                "eye_type", "eye_color", "body_features", "arms_hands", "legs_feet", "wings", "tail",
                "genitals", "clothing", "armor", "tribal_garb", "accessories", "expression", "pose",
                "custom", "style"]

# Wear categories are mutually exclusive under random_all - the creature wears only one of them.
_WEAR_FIELDS = ["clothing", "armor", "tribal_garb"]

# Anatomy categories that get dedicated close-up panels on the creature sheet.
_DETAIL_FIELDS = ["ears", "horns", "mouth", "arms_hands", "legs_feet", "wings", "tail"]


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
    return f"'none' skips this trait, 'random' picks a random entry, 'exclude' keeps it out of the prompt even when random_all is on. Add your own entries in json/star_creature_builder.json (restart ComfyUI to see them in this list, 'random' uses them right away)."


class StarCreatureBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        lists, _, _ = _load_data()
        required = {}
        for field in _LIST_FIELDS:
            options = ["none", "random", "exclude"] + lists.get(field, [])
            default = "fantasy illustration" if field == "style" and "fantasy illustration" in options else "none"
            required[field] = (options, {"default": default, "tooltip": _tooltip(field)})
        required["age"] = ("STRING", {
            "default": "",
            "placeholder": "adult / ancient / random",
            "tooltip": "Free text. Empty = skipped. Type 'random' or enable the random_age toggle for a random age (18-100 by default, change age_random_min/max in the json settings). A plain number becomes 'X year old'."
        })
        required["random_age"] = ("BOOLEAN", {
            "default": False,
            "label_on": "random age (18-100)",
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
            "tooltip": "If enabled, every category gets a random pick regardless of its widget setting. Only one of clothing/armor/tribal_garb is picked. Categories set to 'exclude' stay out of the prompt."
        })
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("description", "creature_sheet", "system_prompt")
    OUTPUT_TOOLTIPS = (
        "The plain generated comma-separated description prompt.",
        "The ready-to-use strict creature reference sheet image prompt for the same creature (multi-panel grid, anatomy close-up panels). Note: an outfit appears if one is selected or randomized - set all wear categories to 'exclude' for a naked sheet.",
        "The system prompt to feed into your LLM so it can write creature sheet prompts itself. Editable as 'system_prompt' in json/star_creature_builder.json.",
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

    def build(self, sex, creature_type, element, body_plan, size, head, ears, horns, mouth,
              skin_type, body_color, pattern, pattern_color, hair_mane, hair_color,
              eye_type, eye_color, body_features, arms_hands, legs_feet, wings, tail,
              genitals, clothing, armor, tribal_garb, accessories, expression, pose,
              custom, style, age, random_age, additional, seed, random_all):
        lists, settings, system_prompt = _load_data()
        if seed == 0:
            seed = random.randint(0, 0xffffffffffffffff)
        rng = random.Random(seed)

        widget_values = dict(sex=sex, creature_type=creature_type, element=element, body_plan=body_plan, size=size,
                             head=head, ears=ears, horns=horns, mouth=mouth,
                             skin_type=skin_type, body_color=body_color, pattern=pattern, pattern_color=pattern_color,
                             hair_mane=hair_mane, hair_color=hair_color,
                             eye_type=eye_type, eye_color=eye_color,
                             body_features=body_features, arms_hands=arms_hands, legs_feet=legs_feet,
                             wings=wings, tail=tail, genitals=genitals,
                             clothing=clothing, armor=armor, tribal_garb=tribal_garb, accessories=accessories,
                             expression=expression, pose=pose, custom=custom, style=style)
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
            age_max = int(settings.get("age_random_max", 100))
            age = str(rng.randint(min(age_min, age_max), max(age_min, age_max)))
        if age and age[0].isdigit():
            age = f"{age} year old"

        skin = ""
        if resolved["body_color"] and resolved["skin_type"]:
            skin = f"{resolved['body_color']} {resolved['skin_type']}"
        elif resolved["skin_type"]:
            skin = resolved["skin_type"]
        elif resolved["body_color"]:
            skin = f"{resolved['body_color']} skin"
        if resolved["pattern"]:
            markings = f"{resolved['pattern_color']} {resolved['pattern']}".strip() if resolved["pattern_color"] else resolved["pattern"]
            skin = f"{skin}, {markings}" if skin else markings
        elif resolved["pattern_color"]:
            skin = f"{skin}, {resolved['pattern_color']} markings" if skin else f"{resolved['pattern_color']} markings"

        head_part = resolved["head"]
        if head_part and resolved["mouth"]:
            head_part = f"{head_part} with {resolved['mouth']}"
        elif resolved["mouth"]:
            head_part = resolved["mouth"]

        hair = ""
        hair_style, color = resolved["hair_mane"], resolved["hair_color"]
        if hair_style and color:
            if "hair" in hair_style:
                hair = hair_style.replace("hair", f"{color} hair", 1)
            elif "mane" in hair_style:
                hair = hair_style.replace("mane", f"{color} mane", 1)
            else:
                hair = f"{color} {hair_style}"
        elif hair_style:
            hair = hair_style
        elif color:
            hair = f"{color} hair"

        eyes = ""
        e_type, e_color = resolved["eye_type"], resolved["eye_color"]
        if e_type and e_color:
            eyes = e_type.replace("eye", f"{e_color} eye", 1) if "eye" in e_type else f"{e_color} {e_type}"
        elif e_type:
            eyes = e_type
        elif e_color:
            eyes = f"{e_color} eyes"

        outfits = [resolved[f] for f in _WEAR_FIELDS]
        outfits = [o for o in outfits if o]
        wear = f"wearing {' and '.join(outfits)}" if outfits else ""

        lead = " ".join(x for x in (resolved["size"], age, resolved["body_plan"], resolved["sex"],
                                    resolved["element"], resolved["creature_type"]) if x)

        sheet_prompt = self._build_sheet(resolved, lead, skin, head_part, hair, eyes, wear, additional)

        parts = []
        if lead:
            parts.append(lead)
        if skin:
            parts.append(skin)
        if head_part:
            parts.append(head_part)
        for f in ("ears", "horns"):
            if resolved[f]:
                parts.append(resolved[f])
        if hair:
            parts.append(hair)
        if eyes:
            parts.append(eyes)
        for f in ("body_features", "arms_hands", "legs_feet", "wings", "tail", "genitals"):
            if resolved[f]:
                parts.append(resolved[f])
        if wear:
            parts.append(wear)
        if resolved["accessories"]:
            parts.append(resolved["accessories"])
        if resolved["expression"]:
            parts.append(resolved["expression"])
        if resolved["pose"]:
            parts.append(resolved["pose"])
        if resolved["custom"]:
            parts.append(resolved["custom"])
        if resolved["style"]:
            parts.append(resolved["style"])
        if additional.strip():
            parts.append(additional.strip())

        return (", ".join(parts), sheet_prompt, system_prompt)

    def _build_sheet(self, resolved, lead, skin, head_part, hair, eyes, wear, additional):
        lead = lead.replace(" year old", "-year-old") if " year old" in lead else lead
        lead = lead or "adult creature"

        traits = [x for x in (skin, head_part, hair, eyes, resolved["body_features"], resolved["custom"]) if x]
        anatomy = [resolved[f] for f in _DETAIL_FIELDS if resolved[f] and not resolved[f].startswith("no ")]
        items = [self._isolate(resolved["accessories"])] if resolved["accessories"] else []

        sentences = []
        article = "an" if lead[0].lower() in "aeiou" else "a"
        first = f"A creature reference sheet of {article} {lead}"
        if traits:
            first += f" with {', '.join(traits)}"
        if anatomy:
            first += f", featuring {', '.join(anatomy)}"
        first += ", presented in a multi-panel grid layout on a solid, neutral background."
        sentences.append(first)

        sentences.append("The left side has two vertical full-body panels spanning the entire height: one full-body front view and one full-body back view.")

        head_row = "the top row shows a frontal head portrait, a side profile portrait, and a 45-degree angle portrait"
        if resolved["expression"]:
            head_row += f" with {resolved['expression']}"
        bottom = "the bottom row shows the chest area from the front, the pelvic area from the front, and the hindquarters from the back"
        if resolved["genitals"] and resolved["genitals"] != "no genitals":
            bottom += f", clearly showing the {resolved['genitals']}"
        sentences.append(f"The right side is a 6-panel detail grid in a 3x2 layout: {head_row}; {bottom}.")

        extra = []
        if anatomy:
            extra.append(f"dedicated anatomical close-ups of the {', '.join(anatomy)}")
        if items:
            extra.append(f"separate isolated close-ups of the following items on neutral backgrounds, never worn or held by the creature: {', '.join(items)}")
        if extra:
            sentences.append(f"The right-side grid expands to 8 panels in a 4x2 layout, adding {' and '.join(extra)}.")

        if wear:
            sentences.append(f"The creature wears {wear.removeprefix('wearing ')} in the full-body panels.")
        else:
            sentences.append("The creature is unclothed in the full-body panels.")

        style = resolved["style"]
        style_sentence = "The style is clinical and scientific, like a biological field guide"
        if style:
            style_sentence = f"Rendered in {style} style, presented in a clinical and scientific way like a biological field guide"
        style_sentence += ", with flat, even studio lighting that clearly illuminates all morphological details without harsh shadows, and enforces absolute consistency in the creature's features, proportions, and overall appearance across every panel."
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
    "StarCreatureBuilder": StarCreatureBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarCreatureBuilder": "⭐ Star Creature Builder",
}
