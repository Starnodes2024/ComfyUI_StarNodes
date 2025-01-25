import os
import random
import re
import folder_paths

class Starwildcards:
    
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'star_wilds'
    CATEGORY = '⭐StarNodes'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_1": ("STRING", {"multiline": True}),
                "prompt_2": ("STRING", {"multiline": True}),
                "prompt_3": ("STRING", {"multiline": True}),
                "prompt_4": ("STRING", {"multiline": True}),
                "prompt_5": ("STRING", {"multiline": True}),
                "prompt_6": ("STRING", {"multiline": True}),
                "prompt_7": ("STRING", {"multiline": True}),
            }
        }
    
    def process(self, seed, prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7):
        # Process each prompt individually
        processed_1 = process_wildcard_syntax(prompt_1, seed)
        processed_2 = process_wildcard_syntax(prompt_2, seed+144)
        processed_3 = process_wildcard_syntax(prompt_3, seed+245)
        processed_4 = process_wildcard_syntax(prompt_4, seed+283)
        processed_5 = process_wildcard_syntax(prompt_5, seed+483)
        processed_6 = process_wildcard_syntax(prompt_6, seed+747)
        processed_7 = process_wildcard_syntax(prompt_7, seed-969)
        
        # Join all processed prompts with spaces
        final_text = " ".join([processed_1, processed_2, processed_3, processed_4, 
                             processed_5, processed_6, processed_7])
        return (final_text,)

def find_and_replace_wildcards(prompt, offset_seed, debug=False):
    # Split the prompt into parts based on wildcards, including potential folder paths
    # This pattern matches folder paths followed by wildcards: ([^_\\]+\\)?(__[^_]*?__)
    parts = re.split(r'((?:[^_\\]+\\)?(?:__[^_]*?__))', prompt)
    
    # Process each part
    result = ""
    wildcard_count = 0  # Counter for wildcards processed
    
    for part in parts:
        # Skip empty parts
        if not part:
            continue
            
        # Check if this part contains a wildcard
        wildcard_match = re.search(r'((?:[^_\\]+\\)?)((?:__[^_]*?__))', part)
        if wildcard_match:
            folder_path = wildcard_match.group(1) or ''  # This will include the folder path if it exists
            wildcard = wildcard_match.group(2)     # This will be just the wildcard
            
            # Get the wildcard name without the __
            wildcard_name = wildcard[2:-2]
            
            # Build the full path
            if folder_path:
                # Remove trailing backslash from folder path
                folder_path = folder_path.rstrip('\\')
                wildcard_path = os.path.join(folder_paths.base_path, 'wildcards', folder_path, wildcard_name + '.txt')
            else:
                wildcard_path = os.path.join(folder_paths.base_path, 'wildcards', wildcard_name + '.txt')
            
            # Check if the file exists
            if os.path.exists(wildcard_path):
                try:
                    # Read the lines from the file
                    with open(wildcard_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if lines:  # Only process if we have lines
                        # Use a different seed for each wildcard by adding the counter
                        current_seed = offset_seed + wildcard_count
                        random.seed(current_seed)
                        selected_line = random.choice(lines)
                        wildcard_count += 1  # Increment counter after processing
                        result += selected_line
                    else:
                        # File exists but is empty, use wildcard name as fallback
                        result += wildcard_name
                except Exception as e:
                    result += wildcard_name
            else:
                # If the file doesn't exist, just use the wildcard name
                result += wildcard_name
        else:
            # Not a wildcard, just add it to the result
            result += part
    
    return result

def process_random_options(text, seed):
    # Use regex to find patterns like {option1|option2|option3} or {__wildcard1__|__wildcard2__}
    random.seed(seed)
    def replace_options(match):
        options = [opt.strip() for opt in match.group(1).split('|')]
        selected = random.choice(options)
        
        # Check if the selected option is a wildcard
        if selected.startswith('__') and selected.endswith('__'):
            # Process the wildcard using the existing functionality
            # We add 1 to the seed to ensure it's different from the main seed
            processed = find_and_replace_wildcards(selected, seed + 1)
            return processed
        return selected
    
    # Replace all occurrences of {options} with a random choice
    processed = re.sub(r'\{([^{}]+)\}', replace_options, text)
    return processed

def process_wildcard_syntax(text, seed):
    # First process the random options in curly braces
    text = process_random_options(text, seed)
    # Then process the wildcards
    processed_text = find_and_replace_wildcards(text, seed)
    return processed_text

def search_and_replace(text):
    # This is a simplified version that just processes wildcards
    return text

def strip_all_syntax(text):
    # Remove all special syntax from the text
    return text

NODE_CLASS_MAPPINGS = {
    "StarFiveWildcards": Starwildcards
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFiveWildcards": "⭐Star Seven Wildcards"
}