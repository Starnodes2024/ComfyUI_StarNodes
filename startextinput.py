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
                "separator": ("STRING", {"default": " "})
            },
            "optional": {
                # First 3 inputs as connection points
                "text1": ("STRING",),
                "text2": ("STRING",),
                "text3": ("STRING",),
                # Last 4 inputs as text fields
                "text4": ("STRING", {"default": ""}),
                "text5": ("STRING", {"default": ""}),
                "text6": ("STRING", {"default": ""}),
                "text7": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "join_texts"
    CATEGORY = "⭐StarNodes"

    def join_texts(self, separator, text1="", text2="", text3="", text4="", text5="", text6="", text7=""):
        # Filter out empty strings
        texts = [t for t in [text1, text2, text3, text4, text5, text6, text7] if t]
        
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
    "StarTextInput": "⭐ Star Seven Inputs (txt)"
}
