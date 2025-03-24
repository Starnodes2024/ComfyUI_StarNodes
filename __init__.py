from .ollamahelper import NODE_CLASS_MAPPINGS as OLLAMA_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OLLAMA_NODE_DISPLAY_NAMES
from .FluxStart import NODE_CLASS_MAPPINGS as FLUXSTART_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FLUXSTART_NODE_DISPLAY_NAMES
from .SDXLStart import NODE_CLASS_MAPPINGS as SDXLSTART_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SDXLSTART_NODE_DISPLAY_NAMES
from .SD35Start import NODE_CLASS_MAPPINGS as SD35START_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SD35START_NODE_DISPLAY_NAMES
from .starupscale import NODE_CLASS_MAPPINGS as STARUPSCALE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARUPSCALE_NODE_DISPLAY_NAMES
from .starwilds import NODE_CLASS_MAPPINGS as STARWILDS_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARWILDS_NODE_DISPLAY_NAMES
from .starwildsadv import NODE_CLASS_MAPPINGS as STARWILDSADV_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARWILDSADV_NODE_DISPLAY_NAMES
from .fluxstarsampler import NODE_CLASS_MAPPINGS as FLUXSTARSAMPLER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FLUXSTARSAMPLER_NODE_DISPLAY_NAMES
from .sdstarsampler import NODE_CLASS_MAPPINGS as SDSTARSAMPLER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SDSTARSAMPLER_NODE_DISPLAY_NAMES
from .StarFluxFiller import NODE_CLASS_MAPPINGS as STARFLUXFILLER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARFLUXFILLER_NODE_DISPLAY_NAMES
from .detailstardaemon import NODE_CLASS_MAPPINGS as DETAILSTARDAEMON_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DETAILSTARDAEMON_NODE_DISPLAY_NAMES
from .starlatentinput import NODE_CLASS_MAPPINGS as STARLATENTINPUT_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARLATENTINPUT_NODE_DISPLAY_NAMES
from .StarNode import NODE_CLASS_MAPPINGS as STARNODE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARNODE_NODE_DISPLAY_NAMES
from .startextfilter import NODE_CLASS_MAPPINGS as STARTEXTFILTER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARTEXTFILTER_NODE_DISPLAY_NAMES
from .startextinput import NODE_CLASS_MAPPINGS as STARTEXTINPUT_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARTEXTINPUT_NODE_DISPLAY_NAMES
from .starfaceloader import NODE_CLASS_MAPPINGS as STARFACELOADER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARFACELOADER_NODE_DISPLAY_NAMES
from .StarDivisibleDimension import NODE_CLASS_MAPPINGS as STARDIVISIBLEDIMENSION_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARDIVISIBLEDIMENSION_NODE_DISPLAY_NAMES
from .StarNewsScraper import NODE_CLASS_MAPPINGS as STARNEWS_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARNEWS_NODE_DISPLAY_NAMES
from .StarLora import NODE_CLASS_MAPPINGS as STARLORA_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARLORA_NODE_DISPLAY_NAMES
from .StarPSDSaver import NODE_CLASS_MAPPINGS as STARPSDSAVER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARPSDSAVER_NODE_DISPLAY_NAMES
from .startextstorage import NODE_CLASS_MAPPINGS as STARTEXTSTORAGE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARTEXTSTORAGE_NODE_DISPLAY_NAMES
from .StarDenoiseSlider import NODE_CLASS_MAPPINGS as STARDENOISESLIDER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARDENOISESLIDER_NODE_DISPLAY_NAMES
from .starsamplersettings_nodes import NODE_CLASS_MAPPINGS as STARSAMPLERSETTINGS_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARSAMPLERSETTINGS_NODE_DISPLAY_NAMES

import os
import shutil
import sys

# Try to import folder_paths from ComfyUI
try:
    import folder_paths
except ImportError:
    print("Warning: Could not import folder_paths from ComfyUI")
    folder_paths = None

NODE_CLASS_MAPPINGS = {
    **OLLAMA_NODE_MAPPINGS,
    **FLUXSTART_NODE_MAPPINGS,
    **SDXLSTART_NODE_MAPPINGS,
    **SD35START_NODE_MAPPINGS,
    **STARUPSCALE_NODE_MAPPINGS,
    **STARWILDS_NODE_MAPPINGS,
    **STARWILDSADV_NODE_MAPPINGS,
    **FLUXSTARSAMPLER_NODE_MAPPINGS,
    **SDSTARSAMPLER_NODE_MAPPINGS,
    **STARFLUXFILLER_NODE_MAPPINGS,
    **STARNEWS_NODE_MAPPINGS,
    **STARTEXTFILTER_NODE_MAPPINGS,
    **DETAILSTARDAEMON_NODE_MAPPINGS,
    **STARNODE_NODE_MAPPINGS,
    **STARTEXTINPUT_NODE_MAPPINGS,
    **STARFACELOADER_NODE_MAPPINGS,
    **STARLATENTINPUT_NODE_MAPPINGS,
    **STARDIVISIBLEDIMENSION_NODE_MAPPINGS,
    **STARLORA_NODE_MAPPINGS,
    **STARPSDSAVER_NODE_MAPPINGS,
    **STARTEXTSTORAGE_NODE_MAPPINGS,
    **STARDENOISESLIDER_NODE_MAPPINGS,
    **STARSAMPLERSETTINGS_NODE_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **OLLAMA_NODE_DISPLAY_NAMES,
    **FLUXSTART_NODE_DISPLAY_NAMES,
    **SDXLSTART_NODE_DISPLAY_NAMES,
    **SD35START_NODE_DISPLAY_NAMES,
    **STARUPSCALE_NODE_DISPLAY_NAMES,
    **STARWILDS_NODE_DISPLAY_NAMES,
    **STARWILDSADV_NODE_DISPLAY_NAMES,
    **FLUXSTARSAMPLER_NODE_DISPLAY_NAMES,
    **SDSTARSAMPLER_NODE_DISPLAY_NAMES,
    **STARFLUXFILLER_NODE_DISPLAY_NAMES,
    **STARNEWS_NODE_DISPLAY_NAMES,
    **STARTEXTFILTER_NODE_DISPLAY_NAMES,
    **DETAILSTARDAEMON_NODE_DISPLAY_NAMES,
    **STARNODE_NODE_DISPLAY_NAMES,
    **STARTEXTINPUT_NODE_DISPLAY_NAMES,
    **STARFACELOADER_NODE_DISPLAY_NAMES,
    **STARLATENTINPUT_NODE_DISPLAY_NAMES,
    **STARDIVISIBLEDIMENSION_NODE_DISPLAY_NAMES,
    **STARLORA_NODE_DISPLAY_NAMES,
    **STARPSDSAVER_NODE_DISPLAY_NAMES,
    **STARTEXTSTORAGE_NODE_DISPLAY_NAMES,
    **STARDENOISESLIDER_NODE_DISPLAY_NAMES,
    **STARSAMPLERSETTINGS_NODE_DISPLAY_NAMES
}

__version__ = "1.3.2"

# Define the web directory for ComfyUI to find our JavaScript files
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Define custom types for ComfyUI
def get_custom_types():
    return {
        "SDSTAR_SETTINGS": {"description": "Settings for SDstarsampler"},
        "FLUXSTAR_SETTINGS": {"description": "Settings for Fluxstarsampler"}
    }

# Add custom types to ComfyUI
try:
    import comfy.custom_types
    comfy.custom_types.ensure_custom_types(get_custom_types())
except ImportError:
    print("Warning: Could not register custom types for StarNodes settings")

# Copy wildcards folder to ComfyUI main directory if it doesn't exist
def copy_wildcards_folder():
    if folder_paths is None:
        print("Warning: Could not copy wildcards folder because folder_paths is not available")
        return
    
    # Get the path to the main ComfyUI directory
    comfyui_base_path = folder_paths.base_path
    
    # Get the path to the wildcards folder in the StarNodes extension
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_wildcards_path = os.path.join(current_dir, 'wildcards')
    
    # Get the path to the destination wildcards folder in the main ComfyUI directory
    dest_wildcards_path = os.path.join(comfyui_base_path, 'wildcards')
    
    # Check if the wildcards folder already exists in the main ComfyUI directory
    if not os.path.exists(dest_wildcards_path):
        try:
            print(f"StarNodes: Copying wildcards folder to {dest_wildcards_path}")
            # Create the destination directory if it doesn't exist
            os.makedirs(dest_wildcards_path, exist_ok=True)
            
            # Copy all files from the source wildcards folder to the destination
            for item in os.listdir(source_wildcards_path):
                source_item = os.path.join(source_wildcards_path, item)
                dest_item = os.path.join(dest_wildcards_path, item)
                
                # Skip if the destination file already exists
                if os.path.exists(dest_item):
                    continue
                
                if os.path.isfile(source_item):
                    shutil.copy2(source_item, dest_item)
                    print(f"StarNodes: Copied wildcard file {item}")
            
            print("StarNodes: Successfully copied wildcards folder to ComfyUI main directory")
        except Exception as e:
            print(f"StarNodes: Error copying wildcards folder: {str(e)}")
    else:
        # Check if we need to copy any missing wildcard files
        for item in os.listdir(source_wildcards_path):
            source_item = os.path.join(source_wildcards_path, item)
            dest_item = os.path.join(dest_wildcards_path, item)
            
            # Skip if the destination file already exists
            if os.path.exists(dest_item):
                continue
            
            if os.path.isfile(source_item):
                try:
                    shutil.copy2(source_item, dest_item)
                    print(f"StarNodes: Copied missing wildcard file {item}")
                except Exception as e:
                    print(f"StarNodes: Error copying wildcard file {item}: {str(e)}")

# Run the copy function when the module is imported
copy_wildcards_folder()
