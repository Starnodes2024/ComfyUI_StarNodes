class StarTextFilter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "filter_type": (["remove_empty_lines", "remove_whitespace", "strip_lines", "remove_between_words"], ),
                "start_word": ("STRING", {"default": "INPUT"}),
                "end_word": ("STRING", {"default": "INPUT"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "filter_text"
    CATEGORY = "⭐StarNodes"

    def filter_text(self, text, filter_type, start_word, end_word):
        if filter_type == "remove_empty_lines":
            result = "\n".join([line for line in text.split("\n") if line.strip()])
        elif filter_type == "remove_whitespace":
            result = "".join(text.split())
        elif filter_type == "strip_lines":
            result = "\n".join([line.strip() for line in text.split("\n")])
        elif filter_type == "remove_between_words":
            import re
            pattern = re.escape(start_word) + r'.*?' + re.escape(end_word)
            result = re.sub(pattern, '', text, flags=re.DOTALL)
        else:
            result = text
            
        return (result,)

NODE_CLASS_MAPPINGS = {
    "StarTextFilter": StarTextFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTextFilter": "⭐Star Text Filter"
}
