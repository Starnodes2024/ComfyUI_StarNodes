import os
import json
import copy

class StarEasyTextStorage:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    # Class variable to track when the storage has been modified
    # This will be used to force ComfyUI to refresh the node
    storage_version = 0
    
    def __init__(self):
        # Path to the storage file within the nodes folder
        self.nodes_storage_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startext.json")
        
        # Use the same storage file for all operations so everything is kept in text_io/startext.json
        self.main_storage_file = self.nodes_storage_file
        
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        """Create the storage files if they don't exist."""
        # Ensure nodes folder storage exists
        if not os.path.exists(self.nodes_storage_file):
            with open(self.nodes_storage_file, 'w') as f:
                json.dump({}, f, indent=2)
        
        # Ensure main ComfyUI folder storage exists
        if not os.path.exists(self.main_storage_file):
            with open(self.main_storage_file, 'w') as f:
                json.dump({}, f, indent=2)
    
    def _load_storage(self):
        """Load the text storage database from both locations."""
        combined_storage = {}
        
        # Load from nodes folder first (legacy storage)
        try:
            with open(self.nodes_storage_file, 'r') as f:
                nodes_storage = json.load(f)
                combined_storage.update(nodes_storage)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create a new one
            with open(self.nodes_storage_file, 'w') as f:
                json.dump({}, f, indent=2)
        
        # Then load from main ComfyUI folder (takes precedence)
        try:
            with open(self.main_storage_file, 'r') as f:
                main_storage = json.load(f)
                combined_storage.update(main_storage)  # This will overwrite any duplicates
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create a new one
            with open(self.main_storage_file, 'w') as f:
                json.dump({}, f, indent=2)
        
        return combined_storage
    
    def _save_storage(self, data):
        """Save data to the text storage database in the main ComfyUI folder."""
        # For saving, we only use the main ComfyUI folder
        with open(self.main_storage_file, 'w') as f:
            json.dump(data, f, indent=2)
        # Increment the storage version to force a refresh of the node
        StarEasyTextStorage.storage_version += 1
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get the list of saved text names for the dropdown
        instance = cls()
        saved_texts = list(instance._load_storage().keys())
        
        # If no texts are saved yet, provide a default option
        if not saved_texts:
            saved_texts = ["No texts saved yet"]
        
        return {
            "required": {
                "mode": (["Save Text", "Load Text", "Remove Text", "Replace Text"], {"default": "Load Text"}),
                "Save-Name": ("STRING", {"default": "My Text"}),
            },
            "optional": {
                "Text-Selector": (saved_texts, {"default": saved_texts[0] if saved_texts else ""}),
                "text_content": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        
    # Define which inputs are required based on the mode
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Get the mode from the inputs
        mode = kwargs.get("mode", "Load Text")
        
        # In Save Text mode, we don't need to validate the Text-Selector
        if mode == "Save Text":
            return True
            
        # For other modes, we need to validate the Text-Selector if it's provided
        return True  # Always return True to allow the operation to proceed

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "process_text"
    CATEGORY = "⭐StarNodes/Text And Data"

    def process_text(self, mode, **kwargs):
        # Extract parameters from kwargs
        save_name = kwargs.get("Save-Name", "")
        text_content = kwargs.get("text_content", "")
        text_selector = kwargs.get("Text-Selector", "")
        
        # In Save Text mode, we ignore the text_selector value
        if mode == "Save Text":
            return self._save_text(save_name, text_content)
        elif mode == "Load Text":
            # If the user selected from the dropdown, use that name instead
            if text_selector and text_selector != "No texts saved yet":
                save_name = text_selector
            return self._load_text(save_name)
        elif mode == "Replace Text":
            # If the user selected from the dropdown, use that name instead
            if text_selector and text_selector != "No texts saved yet":
                save_name = text_selector
            return self._replace_text(save_name, text_content)
        else:  # Remove Text
            # If the user selected from the dropdown, use that name instead
            if text_selector and text_selector != "No texts saved yet":
                save_name = text_selector
            return self._remove_text(save_name)
    
    def _get_unique_name(self, base_name, storage):
        """Generate a unique name by adding a number if the name already exists."""
        if base_name not in storage:
            return base_name
        
        counter = 1
        while f"{base_name}_{counter}" in storage:
            counter += 1
        
        return f"{base_name}_{counter}"
    
    def _save_text(self, text_name, text_content):
        """Save text to storage in the main ComfyUI folder."""
        if not text_name.strip():
            return ("Error: Please provide a name for your text.",)
        
        # Load combined storage to check for existing names
        combined_storage = self._load_storage()
        
        # Generate a unique name if this name already exists
        original_name = text_name
        text_name = self._get_unique_name(text_name, combined_storage)
        
        # Now load only the main storage for saving
        try:
            with open(self.main_storage_file, 'r') as f:
                main_storage = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            main_storage = {}
        
        # Save the text to the main storage
        main_storage[text_name] = text_content
        with open(self.main_storage_file, 'w') as f:
            json.dump(main_storage, f, indent=2)
        
        # Increment version to force refresh
        StarEasyTextStorage.storage_version += 1
        
        # Return appropriate message based on whether the name was changed
        if text_name != original_name:
            return (f"Text saved as '{text_name}' (original name '{original_name}' was already taken).",)
        else:
            return (f"Text '{text_name}' saved successfully.",)
    
    def _load_text(self, text_name):
        """Load text from storage."""
        combined_storage = self._load_storage()
        
        if text_name in combined_storage:
            return (combined_storage[text_name],)
        else:
            available_texts = list(combined_storage.keys())
            if available_texts:
                text_list = ", ".join(available_texts)
                return (f"Text '{text_name}' not found. Available texts: {text_list}",)
            else:
                return ("No texts found in storage. Please save some texts first.",)
    
    def _remove_text(self, text_name):
        """Remove text from storage."""
        combined_storage = self._load_storage()
        
        if text_name not in combined_storage:
            available_texts = list(combined_storage.keys())
            if available_texts:
                text_list = ", ".join(available_texts)
                return (f"Text '{text_name}' not found. Available texts: {text_list}",)
            else:
                return ("No texts found in storage to remove.",)
        
        # Check if the text exists in the main storage
        try:
            with open(self.main_storage_file, 'r') as f:
                main_storage = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            main_storage = {}
        
        if text_name in main_storage:
            # Remove from main storage
            del main_storage[text_name]
            with open(self.main_storage_file, 'w') as f:
                json.dump(main_storage, f, indent=2)
            StarEasyTextStorage.storage_version += 1
            return (f"Text '{text_name}' removed successfully.",)
        
        # If not in main storage, check nodes storage
        try:
            with open(self.nodes_storage_file, 'r') as f:
                nodes_storage = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            nodes_storage = {}
        
        if text_name in nodes_storage:
            # Remove from nodes storage
            del nodes_storage[text_name]
            with open(self.nodes_storage_file, 'w') as f:
                json.dump(nodes_storage, f, indent=2)
            StarEasyTextStorage.storage_version += 1
            return (f"Text '{text_name}' removed from nodes storage successfully.",)
        
        return (f"Error: Text '{text_name}' found in combined storage but not in individual storages.",)
                
    def _replace_text(self, text_name, text_content):
        """Replace text in storage with new content."""
        if not text_name.strip():
            return ("Error: Please provide a name for the text to replace.",)
            
        if not text_content.strip():
            return ("Error: Please provide new content to replace with.",)
        
        combined_storage = self._load_storage()
        
        if text_name not in combined_storage:
            available_texts = list(combined_storage.keys())
            if available_texts:
                text_list = ", ".join(available_texts)
                return (f"Text '{text_name}' not found. Available texts: {text_list}",)
            else:
                return ("No texts found in storage to replace. Please save some texts first.",)
        
        # Check if the text exists in the main storage
        try:
            with open(self.main_storage_file, 'r') as f:
                main_storage = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            main_storage = {}
        
        # Always save to main storage, even if it was originally in nodes storage
        main_storage[text_name] = text_content
        with open(self.main_storage_file, 'w') as f:
            json.dump(main_storage, f, indent=2)
        
        StarEasyTextStorage.storage_version += 1
        return (f"Text '{text_name}' replaced successfully.",)

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        # Return the current storage version to force a refresh when storage changes
        return cls.storage_version

NODE_CLASS_MAPPINGS = {
    "StarEasyTextStorage": StarEasyTextStorage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTextStorage": "⭐StarNodes/Text & Prompt Generation/Easy-Text-Storage",
    "StarEasyTextStorage": "⭐ Star Easy-Text-Storage"
}
