# Release 2.3.8 - Files Included

## Root Files
- `__init__.py` - Updated to version 2.3.8, fixed PiD upscaler registration
- `README.md` - Updated version number to 2.3.8
- `pyproject.toml` - Updated version number to 2.3.8
- `RELEASE_NOTES_2.3.8.md` - Detailed release notes

## Image Tools (`image_tools/`)
- `star_tiled_pid_upscaler.py` - Hidden widgets (degrade_sigma, sigmas, color_bias_fix, model_factor), fixed model factor to 4x
- `star_image_compare.py` - Interactive before/after image comparison node

## Samplers (`samplers/`)
- `star_sampler.py` - Star Sampler (unified) - included for reference

## LTX Video (`ltx_video/`)
- `ltxv_sulphur_aio.py` - Added model_override and override_audio optional inputs

## Video Tools (`video_tools/`)
- `star_video_compressor.py` - Added drop_first_frames and drop_last_frames inputs with audio sync

## Web Documentation (`web/docs/`)
- `StarTiledPiDUpscaler.md` - **NEW** - Comprehensive documentation for PiD upscaler
- `StarLTXVSulphurAllInOne.md` - Updated with model_override and override_audio documentation
- `StarVideoCompressor.md` - Updated with drop_first_frames and drop_last_frames documentation

## Web JavaScript (`web/js/`)
- `star_image_compare.js` - JavaScript for interactive image comparison widget

## Summary of Changes

### Bug Fixes
1. **Star Tiled PiD Upscaler Registration** - Fixed missing NODE_CLASS_MAPPINGS in __init__.py

### New Features
1. **Star Tiled PiD Upscaler** - Hidden advanced widgets, fixed model factor to 4x, added documentation
2. **Star LTXV All-In-One** - Added model_override and override_audio optional inputs
3. **Star Video Compressor** - Added frame dropping from start/end with audio sync

### Documentation
1. Created comprehensive StarTiledPiDUpscaler.md documentation
2. Updated StarLTXVSulphurAllInOne.md with new inputs
3. Updated StarVideoCompressor.md with frame dropping features

## Installation Instructions
1. Backup your current `comfyui_starnodes` folder
2. Copy all files from this release folder to your `custom_nodes/comfyui_starnodes` directory
3. Restart ComfyUI
4. The Star Tiled PiD Upscaler should now appear in the node menu under ⭐StarNodes/Image And Latent
