import os
import random
import re
import folder_paths

class StarwildcardsAdvanced:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'star_wilds_adv'
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
                "prompt_1": ("STRING", {"multiline": True}),
                "wildcard_1": (wildcard_options,),
                "prompt_2": ("STRING", {"multiline": True}),
                "wildcard_2": (wildcard_options,),
                "prompt_3": ("STRING", {"multiline": True}),
                "wildcard_3": (wildcard_options,),
                "prompt_4": ("STRING", {"multiline": True}),
                "wildcard_4": (wildcard_options,),
                "prompt_5": ("STRING", {"multiline": True}),
                "wildcard_5": (wildcard_options,),
                "prompt_6": ("STRING", {"multiline": True}),
                "wildcard_6": (wildcard_options,),
                "prompt_7": ("STRING", {"multiline": True}),
                "wildcard_7": (wildcard_options,),
            }
        }
    
    def star_wilds_adv(self, seed, 
                      prompt_1, wildcard_1, 
                      prompt_2, wildcard_2, 
                      prompt_3, wildcard_3, 
                      prompt_4, wildcard_4, 
                      prompt_5, wildcard_5, 
                      prompt_6, wildcard_6, 
                      prompt_7, wildcard_7):
        # Process each prompt individually
        processed_1 = process_wildcard_syntax(prompt_1, seed)
        processed_2 = process_wildcard_syntax(prompt_2, seed+144)
        processed_3 = process_wildcard_syntax(prompt_3, seed+245)
        processed_4 = process_wildcard_syntax(prompt_4, seed+283)
        processed_5 = process_wildcard_syntax(prompt_5, seed+483)
        processed_6 = process_wildcard_syntax(prompt_6, seed+747)
        processed_7 = process_wildcard_syntax(prompt_7, seed-969)
        
        # Process wildcards if selected (not "None")
        if wildcard_1 != "None":
            if wildcard_1 == "Random":
                wildcard_1 = self.get_random_wildcard(seed+5)
            wildcard_text = process_wildcard_from_file(wildcard_1, seed+5)
            processed_1 = processed_1 + " " + wildcard_text if processed_1 else wildcard_text
            
        if wildcard_2 != "None":
            if wildcard_2 == "Random":
                wildcard_2 = self.get_random_wildcard(seed+144+5)
            wildcard_text = process_wildcard_from_file(wildcard_2, seed+144+5)
            processed_2 = processed_2 + " " + wildcard_text if processed_2 else wildcard_text
            
        if wildcard_3 != "None":
            if wildcard_3 == "Random":
                wildcard_3 = self.get_random_wildcard(seed+245+5)
            wildcard_text = process_wildcard_from_file(wildcard_3, seed+245+5)
            processed_3 = processed_3 + " " + wildcard_text if processed_3 else wildcard_text
            
        if wildcard_4 != "None":
            if wildcard_4 == "Random":
                wildcard_4 = self.get_random_wildcard(seed+283+5)
            wildcard_text = process_wildcard_from_file(wildcard_4, seed+283+5)
            processed_4 = processed_4 + " " + wildcard_text if processed_4 else wildcard_text
            
        if wildcard_5 != "None":
            if wildcard_5 == "Random":
                wildcard_5 = self.get_random_wildcard(seed+483+5)
            wildcard_text = process_wildcard_from_file(wildcard_5, seed+483+5)
            processed_5 = processed_5 + " " + wildcard_text if processed_5 else wildcard_text
            
        if wildcard_6 != "None":
            if wildcard_6 == "Random":
                wildcard_6 = self.get_random_wildcard(seed+747+5)
            wildcard_text = process_wildcard_from_file(wildcard_6, seed+747+5)
            processed_6 = processed_6 + " " + wildcard_text if processed_6 else wildcard_text
            
        if wildcard_7 != "None":
            if wildcard_7 == "Random":
                wildcard_7 = self.get_random_wildcard(seed-969+5)
            wildcard_text = process_wildcard_from_file(wildcard_7, seed-969+5)
            processed_7 = processed_7 + " " + wildcard_text if processed_7 else wildcard_text
        
        # Join all processed prompts with spaces
        final_text = " ".join(filter(None, [processed_1, processed_2, processed_3, processed_4, 
                                          processed_5, processed_6, processed_7]))
        return (final_text,)
        
    def get_random_wildcard(self, seed):
        """Select a random wildcard file from the wildcards folder"""
        wildcards_path = os.path.join(folder_paths.base_path, 'wildcards')
        wildcard_files = []
        
        if os.path.exists(wildcards_path):
            for file in os.listdir(wildcards_path):
                if file.endswith('.txt'):
                    wildcard_files.append(file[:-4])  # Remove the .txt extension
        
        if wildcard_files:
            random.seed(seed)
            return random.choice(wildcard_files)
        else:
            return "None"  # Fallback if no wildcards are found

def process_wildcard_from_file(wildcard_name, seed):
    """Process a wildcard directly from its file"""
    wildcard_path = os.path.join(folder_paths.base_path, 'wildcards', wildcard_name + '.txt')
    
    if os.path.exists(wildcard_path):
        try:
            # Read the lines from the file
            with open(wildcard_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if lines:  # Only process if we have lines
                # Use the seed for randomization
                random.seed(seed)
                selected_line = random.choice(lines)
                
                # Process the selected line for any nested wildcards
                processed_line = process_wildcard_syntax(selected_line, seed)
                return processed_line
            else:
                # File exists but is empty
                return wildcard_name
        except Exception as e:
            # Error reading the file
            return wildcard_name
    else:
        # File doesn't exist
        return wildcard_name

def find_and_replace_wildcards(prompt, offset_seed, debug=False, recursion_depth=0):
    # Prevent infinite recursion
    if recursion_depth > 10:  # Maximum recursion depth
        return prompt
        
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
                        
                        # Process any nested wildcards in the selected line
                        if '__' in selected_line:
                            selected_line = find_and_replace_wildcards(
                                selected_line, 
                                current_seed, 
                                debug,
                                recursion_depth + 1
                            )
                            
                        wildcard_count += 1  # Increment counter after processing
                        result += selected_line
                    else:
                        # File exists but is empty, use wildcard name as fallback
                        result += wildcard_name
                except Exception as e:
                    if debug:
                        print(f"Error processing wildcard {wildcard_name}: {str(e)}")
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
    "StarWildcardsAdvanced": StarwildcardsAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarWildsAdv": "⭐StarNodes/Text & Prompt Generation/Wildcards Advanced",
    "StarWildcardsAdvanced": "⭐ Star Wildcards Advanced"
}
