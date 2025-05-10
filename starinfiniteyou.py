# Import the Star InfiniteYou implementations - using fixed versions
from .star_infiniteyou_apply import StarApplyInfiniteYou
from .star_infiniteyou_patch_saver import StarInfiniteYouSaver
from .star_infiniteyou_patch import StarInfiniteYouPatch

# Define node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StarInfiniteYouApply": StarApplyInfiniteYou,
    "StarInfiniteYouSaver": StarInfiniteYouSaver,
    "StarInfiniteYouPatch": StarInfiniteYouPatch,
}

# Define display names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarInfiniteYouApply": "⭐StarNodes/InfiniteYou/Apply",
    "StarInfiniteYouSaver": "⭐StarNodes/InfiniteYou/Patch Saver",
    "StarInfiniteYouPatch": "⭐StarNodes/InfiniteYou/Patch Loader",
}
