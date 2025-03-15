from .ollamahelper import NODE_CLASS_MAPPINGS as OLLAMA_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OLLAMA_NODE_DISPLAY_NAMES
from .FluxStart import NODE_CLASS_MAPPINGS as FLUXSTART_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FLUXSTART_NODE_DISPLAY_NAMES
from .SDXLStart import NODE_CLASS_MAPPINGS as SDXLSTART_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SDXLSTART_NODE_DISPLAY_NAMES
from .SD35Start import NODE_CLASS_MAPPINGS as SD35START_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SD35START_NODE_DISPLAY_NAMES
from .starupscale import NODE_CLASS_MAPPINGS as STARUPSCALE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARUPSCALE_NODE_DISPLAY_NAMES
from .starwilds import NODE_CLASS_MAPPINGS as STARWILDS_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as STARWILDS_NODE_DISPLAY_NAMES
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

import os

NODE_CLASS_MAPPINGS = {
    **OLLAMA_NODE_MAPPINGS,
    **FLUXSTART_NODE_MAPPINGS,
    **SDXLSTART_NODE_MAPPINGS,
    **SD35START_NODE_MAPPINGS,
    **STARUPSCALE_NODE_MAPPINGS,
    **STARWILDS_NODE_MAPPINGS,
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
    **STARDENOISESLIDER_NODE_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **OLLAMA_NODE_DISPLAY_NAMES,
    **FLUXSTART_NODE_DISPLAY_NAMES,
    **SDXLSTART_NODE_DISPLAY_NAMES,
    **SD35START_NODE_DISPLAY_NAMES,
    **STARUPSCALE_NODE_DISPLAY_NAMES,
    **STARWILDS_NODE_DISPLAY_NAMES,
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
    **STARDENOISESLIDER_NODE_DISPLAY_NAMES
}

__version__ = "1.1.1"

# Define the web directory for ComfyUI to find our JavaScript files
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
