import os
import folder_paths
from .starwildsadv import StarwildcardsAdvanced, process_wildcard_syntax, process_wildcard_from_file


class Star1Wildcard:
    """
    Single-wildcard companion to Star Wildcards Advanced.
    One free text prompt plus one wildcard dropdown, joined into a single string.
    """
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'star_1_wildcard'
    CATEGORY = '⭐StarNodes/Text And Data'

    @classmethod
    def INPUT_TYPES(s):
        # Get list of wildcards from the wildcards folder
        wildcards_path = os.path.join(folder_paths.base_path, 'wildcards')
        wildcard_files = []

        if os.path.exists(wildcards_path):
            for file in os.listdir(wildcards_path):
                if file.endswith('.txt'):
                    wildcard_files.append(file[:-4])  # Remove the .txt extension

        # Sort the list and add "None" and "Random" as options
        wildcard_files.sort()
        wildcard_options = ["None", "Random"] + wildcard_files

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True}),
                "wildcard": (wildcard_options,),
            }
        }

    def star_1_wildcard(self, seed, prompt, wildcard):
        processed = process_wildcard_syntax(prompt, seed)

        if wildcard != "None":
            if wildcard == "Random":
                wildcard = StarwildcardsAdvanced().get_random_wildcard(seed + 5)
            wildcard_text = process_wildcard_from_file(wildcard, seed + 5)
            processed = processed + " " + wildcard_text if processed else wildcard_text

        return (processed,)


NODE_CLASS_MAPPINGS = {
    "Star1Wildcard": Star1Wildcard
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Star1Wildcard": "⭐ Star 1 Wildcard"
}
