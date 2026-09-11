"""
⭐ Star Image Loader Options

Connect the STAR_METADATA output of "⭐ Star Load Image+".
Shows all metadata entries in a scrollable on-node list (with copy buttons)
and outputs the 5 custom StarMetaData values on their own connectors.
"""

import json
import re


def _as_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class StarImageLoaderOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("STAR_METADATA", {"tooltip": "metadata output of ⭐ Star Load Image+."}),
                "lookup_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional: type any metadata key (or custom field name) and get its value from the lookup_value output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "StarMetaData 1",
        "StarMetaData 2",
        "StarMetaData 3",
        "StarMetaData 4",
        "StarMetaData 5",
        "lookup_value",
        "raw_json",
    )
    OUTPUT_TOOLTIPS = (
        "Custom metadata value 1.",
        "Custom metadata value 2.",
        "Custom metadata value 3.",
        "Custom metadata value 4.",
        "Custom metadata value 5.",
        "Value of the key typed into lookup_key.",
        "All metadata as formatted JSON.",
    )
    FUNCTION = "extract"
    CATEGORY = "⭐StarNodes/IO"
    DESCRIPTION = (
        "Display all metadata stored in an image and output the 5 custom "
        "StarMetaData values for reuse in the workflow."
    )

    def extract(self, metadata, lookup_key=""):
        md = metadata if isinstance(metadata, dict) else {}

        # lookup: exact key, custom field name, case-insensitive, partial match.
        lookup_value = ""
        key = (lookup_key or "").strip()
        if key:
            if key in md:
                lookup_value = _as_str(md[key])
            else:
                lowered = key.lower()
                for i in range(1, 6):
                    name = md.get(f"custom_{i}_name")
                    if name and str(name).lower() == lowered:
                        lookup_value = _as_str(md.get(f"StarMetaData {i}"))
                        break
                if not lookup_value:
                    for k, v in md.items():
                        if str(k).lower() == lowered:
                            lookup_value = _as_str(v)
                            break
                    else:
                        for k, v in md.items():
                            if lowered in str(k).lower():
                                lookup_value = _as_str(v)
                                break

        # Display entries (custom fields get their user-given name as label).
        entries = []
        for k, v in md.items():
            k = str(k)
            match = re.fullmatch(r"StarMetaData ([1-5])", k)
            if match:
                name = md.get(f"custom_{match.group(1)}_name")
                label = f"{name} ({k})" if name else k
                entries.append([label, _as_str(v)])
            elif re.fullmatch(r"custom_[1-5]_name", k):
                continue  # helper keys, shown via their parent entry
            else:
                entries.append([k, _as_str(v)])

        raw_json = json.dumps(md, ensure_ascii=False, indent=2)

        return {
            "ui": {"star_metadata": [entries]},
            "result": (
                _as_str(md.get("StarMetaData 1")),
                _as_str(md.get("StarMetaData 2")),
                _as_str(md.get("StarMetaData 3")),
                _as_str(md.get("StarMetaData 4")),
                _as_str(md.get("StarMetaData 5")),
                lookup_value,
                raw_json,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "StarImageLoaderOptions": StarImageLoaderOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarImageLoaderOptions": "⭐ Star Image Loader Options",
}
