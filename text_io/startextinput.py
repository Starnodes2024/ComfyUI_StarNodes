class FlexibleInputs(dict):
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    """A special class to make flexible node inputs."""
    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type, )

    def __contains__(self, key):
        return True

class StarTextInput:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": " "}),
                "num_strings": ("INT", {"default": 2, "min": 1, "max": 64})
            },
            "optional": FlexibleInputs("STRING"),
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "join_texts"
    CATEGORY = "⭐StarNodes/Text And Data"

    def join_texts(self, separator, **kwargs):
        # Collect all inputs whose name starts with "text" and sort by numeric suffix
        items = []
        for key, value in kwargs.items():
            if not key.startswith("text"):
                continue
            try:
                idx = int(key.replace("text", ""))
            except ValueError:
                # Skip any unexpected keys that don't follow text<number>
                continue
            if value:
                items.append((idx, value))

        # Sort by index and extract text values
        items.sort(key=lambda x: x[0])
        texts = [v for _, v in items]

        # If no inputs are provided, return the default message
        if not texts:
            return ("A cute little monster holding a sign with big text: GIVE ME INPUT!",)

        # Join the non-empty texts with the separator
        result = separator.join(texts)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "StarTextInput": StarTextInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTextInput": "⭐ Star Text Inputs (Concatenate)"
}
