"""
⭐ Star Metadata Saver Option

Optional companion node for "⭐ Star Save Image+".
Provides 5 custom metadata fields (StarMetaData 1-5) that get embedded
into every saved image.
"""


class StarMetadataSaverOption:
    @classmethod
    def INPUT_TYPES(cls):
        fields = {}
        for i in range(1, 6):
            fields[f"StarMetaData {i}"] = (
                "STRING",
                {"default": "", "multiline": True, "tooltip": f"Custom metadata value {i} (embedded in saved images)."},
            )
        return {"required": fields}

    RETURN_TYPES = ("STAR_SAVE_OPTIONS",)
    RETURN_NAMES = ("save_options",)
    OUTPUT_TOOLTIPS = ("Connect to the 'options' input of ⭐ Star Save Image+.",)
    FUNCTION = "build_options"
    CATEGORY = "⭐StarNodes/IO"
    DESCRIPTION = (
        "5 custom metadata fields (StarMetaData 1-5) that ⭐ Star Save Image+ "
        "embeds into every saved image (PNG, JPG and WEBP)."
    )

    def build_options(self, **kwargs):
        metadata = {}
        for i in range(1, 6):
            value = kwargs.get(f"StarMetaData {i}") or ""
            if value != "":
                metadata[f"StarMetaData {i}"] = value
        return ({"metadata": metadata},)


NODE_CLASS_MAPPINGS = {
    "StarMetadataSaverOption": StarMetadataSaverOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMetadataSaverOption": "⭐ Star Metadata Saver Option",
}
