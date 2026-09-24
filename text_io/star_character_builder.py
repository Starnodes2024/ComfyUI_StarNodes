import os
import json
import time
import random

_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json", "star_character_builder.json")

# Field order used for both the widget layout and the prompt assembly.
_LIST_FIELDS = ["sex", "type", "ethnicity", "skin_tone", "body_build", "hairstyle", "hair_color", "eye_color", "makeup", "clothing", "female_outfit", "male_outfit", "lingerie", "kinky_outfit", "accessories", "sextoy", "expression", "pose", "custom", "style"]

# Wear categories are mutually exclusive under random_all - the character wears only one of them.
_WEAR_FIELDS = ["clothing", "female_outfit", "male_outfit", "lingerie", "kinky_outfit"]


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
    return f"'none' skips this trait, 'random' picks a random entry, 'exclude' keeps it out of the prompt even when random_all is on. Add your own entries in json/star_character_builder.json (restart ComfyUI to see them in this list, 'random' uses them right away)."


class StarCharacterBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        lists, _, _ = _load_data()
        required = {}
        for field in _LIST_FIELDS:
            options = ["none", "random", "exclude"] + lists.get(field, [])
            default = "photorealistic" if field == "style" and "photorealistic" in options else "none"
            required[field] = (options, {"default": default, "tooltip": _tooltip(field)})
        required["age"] = ("STRING", {
            "default": "",
            "placeholder": "25 / young adult / random",
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
            "tooltip": "If enabled, every category gets a random pick regardless of its widget setting. Only one of clothing/female_outfit/male_outfit/lingerie/kinky_outfit is picked. Categories set to 'exclude' stay out of the prompt."
        })
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("description", "character_sheet", "system_prompt")
    OUTPUT_TOOLTIPS = (
        "The plain generated comma-separated description prompt.",
        "The ready-to-use strict character sheet image prompt for the same character (multi-panel grid, gender-correct anatomy panels). Note: an outfit appears if one is selected or randomized - set all wear categories to 'exclude' for a naked sheet.",
        "The system prompt to feed into your LLM so it can write character sheet prompts itself. Editable as 'system_prompt' in json/star_character_builder.json.",
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

    def build(self, sex, type, ethnicity, skin_tone, body_build, hairstyle, hair_color, eye_color,
              makeup, clothing, female_outfit, male_outfit, lingerie, kinky_outfit, accessories,
              sextoy, expression, pose, custom, style, age, random_age, additional, seed, random_all):
        lists, settings, system_prompt = _load_data()
        if seed == 0:
            seed = random.randint(0, 0xffffffffffffffff)
        rng = random.Random(seed)

        widget_values = dict(sex=sex, type=type, ethnicity=ethnicity, skin_tone=skin_tone, body_build=body_build,
                             hairstyle=hairstyle, hair_color=hair_color, eye_color=eye_color,
                             makeup=makeup, clothing=clothing, female_outfit=female_outfit, male_outfit=male_outfit,
                             lingerie=lingerie, kinky_outfit=kinky_outfit, accessories=accessories,
                             sextoy=sextoy, expression=expression, pose=pose, custom=custom, style=style)
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

        hair = ""
        hair_style, color = resolved["hairstyle"], resolved["hair_color"]
        if hair_style and color:
            hair = hair_style.replace("hair", f"{color} hair", 1) if "hair" in hair_style else f"{color} {hair_style}"
        elif hair_style:
            hair = hair_style
        elif color:
            hair = f"{color} hair"

        eyes = resolved["eye_color"]
        if eyes and "eye" not in eyes:
            eyes = f"{eyes} eyes"

        outfits = [resolved[f] for f in ("clothing", "female_outfit", "male_outfit", "lingerie", "kinky_outfit")]
        outfits = [o for o in outfits if o]
        wear = f"wearing {' and '.join(outfits)}" if outfits else ""

        sheet_prompt = self._build_sheet(resolved, age, hair, eyes, wear, additional, resolved["style"])

        parts = []
        lead = " ".join(x for x in (age, resolved["ethnicity"], resolved["type"], resolved["sex"]) if x)
        if lead:
            parts.append(lead)
        if resolved["skin_tone"]:
            parts.append(f"{resolved['skin_tone']} skin")
        if resolved["body_build"]:
            parts.append(resolved["body_build"])
        if hair:
            parts.append(hair)
        if eyes:
            parts.append(eyes)
        if resolved["makeup"]:
            parts.append(resolved["makeup"])
        if wear:
            parts.append(wear)
        if resolved["accessories"]:
            parts.append(resolved["accessories"])
        if resolved["sextoy"]:
            parts.append(resolved["sextoy"])
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

    def _build_sheet(self, resolved, age, hair, eyes, wear, additional, style):
        age = age.replace(" year old", "-year-old") if " year old" in age else age
        lead = " ".join(x for x in (age, resolved["ethnicity"], resolved["type"], resolved["sex"]) if x) or "adult person"

        face_trait = {"female": "a distinctly feminine female face",
                      "male": "a distinctly masculine male face"}.get(resolved["sex"], "")
        traits = [x for x in (resolved["skin_tone"] and f"{resolved['skin_tone']} skin",
                              resolved["body_build"], hair, eyes, face_trait, resolved["custom"]) if x]

        anatomy = {
            "female": "The grid features distinct uncensored anatomical close-up panels of the breasts and the vulva with clearly visible vagina. The character has exactly one vagina and vulva and absolutely no penis, no scrotum, and no male genitalia anywhere in the sheet, with no hermaphroditic or mixed traits.",
            "male": "The grid features distinct uncensored anatomical close-up panels of the penis with testicles, shown in both a flaccid and a fully erect state. The character has a flat male chest, exactly one penis, and absolutely no breasts, no vagina, and no vulva anywhere in the sheet, with no hermaphroditic or mixed traits.",
            "transgender woman": "The grid features distinct uncensored anatomical close-up panels of the breasts and the penis, the penis shown in both a flaccid and a fully erect state. The character has both breasts and a penis, reflecting a male-to-female transition, with absolutely no vagina and no vulva anywhere in the sheet, and no hermaphroditic or mixed traits.",
            "transgender man": "The grid features distinct uncensored anatomical close-up panels of the chest and the vulva with clearly visible vagina. The character has a flat male chest and one vagina, reflecting a female-to-male transition, with absolutely no penis, no scrotum, and no male genitalia anywhere in the sheet, and no hermaphroditic or mixed traits.",
        }.get(resolved["sex"], "The grid features specific uncensored close-up panels of the chest and the genitalia, depicting exactly one consistent type of genitalia with no hermaphroditic or mixed traits anywhere in the sheet.")

        items = [self._isolate(resolved[f]) for f in ("accessories", "sextoy") if resolved[f]]

        style_sentence = "The style is clinical and scientific"
        if style:
            style_sentence = f"Rendered in {style} style, presented in a clinical and scientific way"
        style_sentence += ", featuring flat, even studio lighting that clearly illuminates all morphological details without harsh shadows, and enforces consistent proportions and anatomy across every panel."

        sentences = []
        first = f"A character reference sheet of a {lead}"
        if traits:
            first += f" with {', '.join(traits)}"
        first += ", presented in a multi-panel grid on a solid neutral background."
        sentences.append(first)
        sentences.append("The layout includes a full-body front view, a full-body back view, a front face close-up, and a profile face close-up.")
        face = [x for x in (resolved["expression"], resolved["makeup"]) if x]
        if face:
            sentences.append(f"The face close-ups show {' and '.join(face)}.")
        if wear:
            sentences.append(f"The character wears {wear.removeprefix('wearing ')} in all main body views.")
        else:
            sentences.append("The character is 100% naked in all main body views.")
        sentences.append(anatomy)
        if items:
            sentences.append(f"The following items appear completely isolated in their own separate panels and are never worn or held by the character: {', '.join(items)}.")
        sentences.append(style_sentence)
        sentences.append("No text, letters, numbers, symbols, logos, watermarks, charts, or labels appear anywhere in the image.")
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
    "StarCharacterBuilder": StarCharacterBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarCharacterBuilder": "⭐ Star Character Builder",
}
