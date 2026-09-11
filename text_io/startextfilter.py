class StarTextFilter:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "filter_type": (["remove_between_words", "remove_before_start_word", "remove_after_end_word", "remove_empty_lines", "remove_whitespace", "strip_lines"], ),
                "start_word": ("STRING", {"default": "INPUT"}),
                "end_word": ("STRING", {"default": "INPUT"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "filter_text"
    CATEGORY = "⭐StarNodes/Text And Data"

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
        elif filter_type == "remove_before_start_word":
            import re
            # Find the first occurrence of start_word
            match = re.search(re.escape(start_word), text)
            if match:
                # Return everything from the start_word to the end
                result = text[match.start():]
            else:
                # If start_word not found, return original text
                result = text
        elif filter_type == "remove_after_end_word":
            import re
            # Find the last occurrence of end_word
            match = re.search(re.escape(end_word), text)
            if match:
                # Return everything from the beginning to the end of end_word
                result = text[:match.end()]
            else:
                # If end_word not found, return original text
                result = text
        else:
            result = text
            
        return (result,)

NODE_CLASS_MAPPINGS = {
    "StarTextFilter": StarTextFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTextFilter": "⭐ Star Text Filter"
}
