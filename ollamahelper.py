import os
import nodes

class OllamaModelChooser:
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically load models for the dropdown
        models_path = os.path.join(os.path.dirname(__file__), "ollamamodels.txt")
        available_models = ["No models found"]
        default_model = "No models found"
        
        try:
            if os.path.exists(models_path):
                with open(models_path, 'r', encoding='utf-8') as f:
                    available_models = [line.strip() for line in f.readlines() if line.strip()]
                
                # Set the first model as the default if models exist
                if available_models:
                    default_model = available_models[0]
        except Exception as e:
            print(f"Error reading models file: {e}")
        
        return {
            "required": {
                "Model": (available_models, {"default": default_model}),
                "Instructions": ("STRING", {
                    "multiline": True, 
                    "default": "You are an art expert in creating stunning image prompts in english language and you use 400 tokens max. Thank you in advance!"
                }),
            },
        }
    RETURN_TYPES = ("STRING", (),)
    RETURN_NAMES = ("Instructions (System)", "Ollama Model",)
    FUNCTION = "select_model"
    CATEGORY = "⭐StarNodes"

    def select_model(self, Model, Instructions):
        return (Instructions, Model)

# Mapping for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "OllamaModelChooser": OllamaModelChooser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaModelChooser": "⭐ Starnode Ollama Helper"
}