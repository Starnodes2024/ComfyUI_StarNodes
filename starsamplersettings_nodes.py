import os
import json
import folder_paths
from .starsamplersettings import StarSamplerSettings

class StarSaveSamplerSettings:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "settings_name": ("STRING", {"default": "My Settings", "multiline": False}),
            },
            "optional": {
                "sdstar_settings": ("SDSTAR_SETTINGS", ),
                "fluxstar_settings": ("FLUXSTAR_SETTINGS", ),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("message", )
    FUNCTION = "save_settings"
    CATEGORY = "⭐StarNodes/Settings"
    
    def save_settings(self, settings_name, sdstar_settings=None, fluxstar_settings=None):
        settings_manager = StarSamplerSettings()
        
        if sdstar_settings is not None:
            # Ensure settings is a dictionary
            if not isinstance(sdstar_settings, dict):
                try:
                    sdstar_settings = dict(sdstar_settings)
                except Exception as e:
                    return (f"Error: Could not save settings - {str(e)}", )
            
            success, message = settings_manager.save_settings(
                settings_name, 
                "sdstar", 
                sdstar_settings
            )
            return (message, )
        
        elif fluxstar_settings is not None:
            # Ensure settings is a dictionary
            if not isinstance(fluxstar_settings, dict):
                try:
                    fluxstar_settings = dict(fluxstar_settings)
                except Exception as e:
                    return (f"Error: Could not save settings - {str(e)}", )
            
            # Ensure all values are JSON serializable
            for key, value in list(fluxstar_settings.items()):
                if not isinstance(value, (int, float, bool, str, type(None))):
                    fluxstar_settings[key] = str(value)
            
            success, message = settings_manager.save_settings(
                settings_name, 
                "fluxstar", 
                fluxstar_settings
            )
            return (message, )
        
        else:
            return ("No settings provided to save.", )


class StarLoadSamplerSettings:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @classmethod
    def INPUT_TYPES(s):
        settings_manager = StarSamplerSettings()
        available_settings = settings_manager.get_all_settings()
        
        return {
            "required": {
                "settings_name": (available_settings, ),
            }
        }
    
    RETURN_TYPES = ("SDSTAR_SETTINGS", "FLUXSTAR_SETTINGS", "STRING")
    RETURN_NAMES = ("sdstar_settings", "fluxstar_settings", "message")
    FUNCTION = "load_settings"
    CATEGORY = "⭐StarNodes/Settings"
    
    def load_settings(self, settings_name):
        settings_manager = StarSamplerSettings()
        settings, message = settings_manager.load_settings(settings_name)
        
        if settings is None:
            return (None, None, message)
        
        if settings["sampler_type"] == "sdstar":
            return (settings["settings"], None, message)
        else:
            return (None, settings["settings"], message)


class StarDeleteSamplerSettings:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @classmethod
    def INPUT_TYPES(s):
        settings_manager = StarSamplerSettings()
        available_settings = settings_manager.get_all_settings()
        
        return {
            "required": {
                "settings_name": (available_settings, ),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("message", )
    FUNCTION = "delete_settings"
    CATEGORY = "⭐StarNodes/Settings"
    
    def delete_settings(self, settings_name):
        settings_manager = StarSamplerSettings()
        success, message = settings_manager.delete_settings(settings_name)
        return (message, )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "StarSaveSamplerSettings": StarSaveSamplerSettings,
    "StarLoadSamplerSettings": StarLoadSamplerSettings,
    "StarDeleteSamplerSettings": StarDeleteSamplerSettings,
}

# Import the actual sampler classes to patch them
from .fluxstarsampler import Fluxstarsampler
from .sdstarsampler import SDstarsampler

# The settings outputs are now directly added to the sampler classes
# This code is no longer needed but kept for backwards compatibility
# with older versions of the samplers
# if "FLUXSTAR_SETTINGS" not in Fluxstarsampler.RETURN_TYPES:
#     Fluxstarsampler.RETURN_TYPES = Fluxstarsampler.RETURN_TYPES + ("FLUXSTAR_SETTINGS",)
#     Fluxstarsampler.RETURN_NAMES = Fluxstarsampler.RETURN_NAMES + ("settings_output",)

# if "SDSTAR_SETTINGS" not in SDstarsampler.RETURN_TYPES:
#     SDstarsampler.RETURN_TYPES = SDstarsampler.RETURN_TYPES + ("SDSTAR_SETTINGS",)
#     SDstarsampler.RETURN_NAMES = SDstarsampler.RETURN_NAMES + ("settings_output",)

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSaveSamplerSettings": "⭐ Star Save Sampler Settings",
    "StarLoadSamplerSettings": "⭐ Star Load Sampler Settings",
    "StarDeleteSamplerSettings": "⭐ Star Delete Sampler Settings",
}
