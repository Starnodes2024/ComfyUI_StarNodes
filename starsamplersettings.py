import os
import json
import copy
from pathlib import Path

class StarSamplerSettings:
    """Utility class to manage sampler settings for StarNodes samplers."""
    
    def __init__(self):
        # Define the paths for settings storage using Path for better path handling
        self.starnodes_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Correctly determine the ComfyUI directory path
        # custom_nodes/comfyui_starnodes -> custom_nodes -> ComfyUI
        self.comfyui_dir = self.starnodes_dir.parent.parent
        
        # Create settings file paths
        self.starnodes_settings_path = self.starnodes_dir / "samplersettings.json"
        self.comfyui_settings_path = self.comfyui_dir / "samplersettings.json"
        
        # Initialize settings storage
        self.settings = self._load_settings()
    
    def _load_settings(self):
        """Load settings from both JSON files and merge them."""
        settings = {}
        
        # Load settings from StarNodes directory
        if self.starnodes_settings_path.exists():
            try:
                with open(self.starnodes_settings_path, 'r') as f:
                    settings.update(json.load(f))
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
        
        # Load settings from ComfyUI main directory
        if self.comfyui_settings_path.exists():
            try:
                with open(self.comfyui_settings_path, 'r') as f:
                    settings.update(json.load(f))
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
        
        return settings
    
    def _save_settings(self):
        """Save settings to both JSON files."""
        # Ensure directories exist
        self.starnodes_dir.mkdir(exist_ok=True)
        self.comfyui_settings_path.parent.mkdir(exist_ok=True)
        
        # Save to StarNodes directory
        try:
            with open(self.starnodes_settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception:
            pass
        
        # Save to ComfyUI main directory (backup)
        try:
            with open(self.comfyui_settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception:
            pass
    
    def get_all_settings(self):
        """Get all available settings names."""
        return list(self.settings.keys())
    
    def save_settings(self, settings_name, sampler_type, settings_dict):
        """Save settings with the given name."""
        if not settings_name:
            return False, "Settings name cannot be empty"
        
        # Validate settings_dict
        if settings_dict is None:
            return False, "Settings dictionary is None"
            
        if not isinstance(settings_dict, dict):
            return False, "Settings is not a dictionary"
        
        # Create a new settings entry
        entry = {
            "sampler_type": sampler_type,
            "settings": settings_dict
        }
        
        # Save to settings dictionary
        self.settings[settings_name] = entry
        
        # Force save to both files
        try:
            self._save_settings()
            return True, f"Settings '{settings_name}' saved successfully"
        except Exception as e:
            return False, f"Error saving settings: {str(e)}"
    
    def load_settings(self, settings_name):
        """Load settings with the given name."""
        if settings_name not in self.settings:
            return None, f"Settings '{settings_name}' not found"
        
        return self.settings[settings_name], f"Settings '{settings_name}' loaded successfully"
    
    def delete_settings(self, settings_name):
        """Delete settings with the given name."""
        if settings_name not in self.settings:
            return False, f"Settings '{settings_name}' not found"
        
        # Remove from settings dictionary
        del self.settings[settings_name]
        self._save_settings()
        
        return True, f"Settings '{settings_name}' deleted successfully"
    
    def extract_sdstar_settings(self, node):
        """Extract settings from an SDstarsampler node."""
        settings = {
            "seed": node.get("seed", 0),
            "control_after_generate": node.get("control_after_generate", False),
            "sampler_name": node.get("sampler_name", "res_2m_sde"),
            "scheduler": node.get("scheduler", "beta57"),
            "steps": node.get("steps", 20),
            "cfg": node.get("cfg", 7.0),
            "denoise": node.get("denoise", 1.0)
        }
        return settings
    
    def extract_fluxstar_settings(self, node):
        """Extract settings from a Fluxstarsampler node."""
        settings = {
            "seed": node.get("seed", 0),
            "control_after_generate": node.get("control_after_generate", False),
            "sampler": node.get("sampler", "res_2m_sde"),
            "scheduler": node.get("scheduler", "beta57"),
            "steps": node.get("steps", "20"),
            "guidance": node.get("guidance", "3.5"),
            "max_shift": node.get("max_shift", "1.15"),
            "base_shift": node.get("base_shift", "0.5"),
            "denoise": node.get("denoise", "1.0"),
            "use_teacache": node.get("use_teacache", True)
        }
        return settings
    
    def apply_settings_to_sdstar(self, node, settings):
        """Apply loaded settings to an SDstarsampler node."""
        if "settings" in settings:
            settings = settings["settings"]
            
        # Apply each setting if it exists in the settings
        if "seed" in settings:
            node["seed"] = settings["seed"]
        if "control_after_generate" in settings:
            node["control_after_generate"] = settings["control_after_generate"]
        if "sampler_name" in settings:
            node["sampler_name"] = settings["sampler_name"]
        if "scheduler" in settings:
            node["scheduler"] = settings["scheduler"]
        if "steps" in settings:
            node["steps"] = settings["steps"]
        if "cfg" in settings:
            node["cfg"] = settings["cfg"]
        if "denoise" in settings:
            node["denoise"] = settings["denoise"]
        
        return node
    
    def apply_settings_to_fluxstar(self, node, settings):
        """Apply loaded settings to a Fluxstarsampler node."""
        if "settings" in settings:
            settings = settings["settings"]
            
        # Apply each setting if it exists in the settings
        if "seed" in settings:
            node["seed"] = settings["seed"]
        if "control_after_generate" in settings:
            node["control_after_generate"] = settings["control_after_generate"]
        if "sampler" in settings:
            node["sampler"] = settings["sampler"]
        if "scheduler" in settings:
            node["scheduler"] = settings["scheduler"]
        if "steps" in settings:
            node["steps"] = settings["steps"]
        if "guidance" in settings:
            node["guidance"] = settings["guidance"]
        if "max_shift" in settings:
            node["max_shift"] = settings["max_shift"]
        if "base_shift" in settings:
            node["base_shift"] = settings["base_shift"]
        if "denoise" in settings:
            node["denoise"] = settings["denoise"]
        if "use_teacache" in settings:
            node["use_teacache"] = settings["use_teacache"]
        
        return node
