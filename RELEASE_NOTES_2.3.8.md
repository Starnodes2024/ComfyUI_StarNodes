# Release Notes - Version 2.3.8

## Summary
This release includes enhancements to existing nodes and bug fixes for proper node registration.

## Changes

### New Features

#### ⭐ Star Tiled PiD Upscaler - Improvements
- **Hidden Advanced Settings**: `degrade_sigma`, `sigmas`, `color_bias_fix`, and `model_factor` widgets are now hidden from the UI
- **Fixed Model Factor**: Model factor is now fixed to 4x for optimal performance
- **New Documentation**: Added comprehensive in-app help documentation at `web/docs/StarTiledPiDUpscaler.md`

### Enhancements

#### ⭐ Star LTXV All-In-One (Sulphur AIO)
- **Model Override**: Added optional `model_override` MODEL input to use externally patched models (flash/sage attention)
- **Override Audio**: Added optional `override_audio` BOOLEAN input to control audio output source
- Updated documentation in `web/docs/StarLTXVSulphurAllInOne.md`

#### ⭐ Star Video Compressor
- **Drop First Frames**: Added `drop_first_frames` INT input (0-1000) to skip frames from start
- **Drop Last Frames**: Added `drop_last_frames` INT input (0-1000) to cut frames from end
- Both features maintain audio sync automatically
- Updated documentation in `web/docs/StarVideoCompressor.md`

### Bug Fixes

#### Node Registration
- **Fixed**: Star Tiled PiD Upscaler now properly registered in `__init__.py`
  - Added missing `**PID_MAPPINGS` and `**PID_DISPLAY_MAPPINGS` to main registration dictionaries
  - Node is now correctly recognized by ComfyUI

### Updated Files

#### Core Files
- `__init__.py` - Version 2.3.8, fixed PiD upscaler registration
- `README.md` - Updated to version 2.3.8
- `pyproject.toml` - Updated to version 2.3.8

#### Image Tools
- `image_tools/star_tiled_pid_upscaler.py` - Hidden widgets, fixed model factor to 4x
- `image_tools/star_image_compare.py` - Included for reference

#### Samplers
- `samplers/star_sampler.py` - Star Sampler (unified) included

#### LTX Video
- `ltx_video/ltxv_sulphur_aio.py` - Added model_override and override_audio inputs

#### Video Tools
- `video_tools/star_video_compressor.py` - Added drop_first_frames and drop_last_frames

#### Web Assets
- `web/docs/StarTiledPiDUpscaler.md` - New comprehensive documentation
- `web/docs/StarLTXVSulphurAllInOne.md` - Updated with new inputs
- `web/docs/StarVideoCompressor.md` - Updated with frame dropping features
- `web/js/star_image_compare.js` - JavaScript for image compare node

## Installation
Copy the contents of this release folder to your ComfyUI custom_nodes/comfyui_starnodes directory, replacing existing files.

## Requirements
- ComfyUI with native PixelDiT/PiD support for Star Tiled PiD Upscaler
- No additional dependencies required for other nodes

## Notes
- The Star Tiled PiD Upscaler now uses fixed 4x model factor for optimal VRAM usage
- Advanced settings are available as keyword defaults in the code but not exposed in the UI
- Audio sync is automatically maintained when using frame dropping in Star Video Compressor
