# Release Notes - ComfyUI StarNodes v2.1.1

**Release Date:** July 10, 2026

## Overview

Version 2.1.1 is an enhancement release that adds interactive controls and fullscreen support to the **Star 360 Parallax Viewer** node, along with critical bug fixes for the panorama viewing experience.

## ✨ Enhancements

### Star 360 Parallax Viewer

The panorama viewer now includes a comprehensive **on-screen control bar** at the bottom of the viewer with the following features:

#### Navigation Controls
- **Pan Arrows (◀ ▲ ▼ ▶)** - Press and hold to continuously pan the view in any direction
- **Reset Button (⌂)** - Return camera to initial orientation and default zoom level (75° FOV)

#### Zoom Controls
- **Zoom Buttons (− / +)** - Manual zoom in/out controls (same 30°-120° range as scroll wheel)
- Scroll wheel zoom still available for convenience

#### Auto-Rotation
- **Play/Pause Toggle (▶/⏸)** - Start or stop automatic horizontal rotation of the panorama
- **Speed Slider** - Adjust auto-rotation speed from -5 to +5
  - Positive values: rotate right
  - Negative values: rotate left (reverse)
  - Zero: pause rotation

#### Fullscreen Mode
- **Fullscreen Button (⛶)** - Toggle fullscreen viewing mode
- Properly handles aspect ratio when entering/exiting fullscreen
- Press Esc or click the button again to exit fullscreen

#### Technical Improvements
- **Improved Mouse Controls** - More responsive drag-to-pan navigation with smoother interpolation
- **Per-Run Cleanup** - Properly cancels animation frames and removes old controls when re-executing the workflow
- **Event Listener Cleanup** - Uses `AbortController` to properly clean up all event listeners between runs, preventing memory leaks

## 🐛 Bug Fixes

### Critical Rendering Fix
- **Fixed Black Screen Issue** - The viewer was showing only a black square after execution
  - Root cause: Double-flip from both `geometry.scale(-1, 1, 1)` and `THREE.BackSide` made the sphere invisible from inside
  - Solution: Changed material side to `THREE.FrontSide` to work correctly with the inverted geometry
  - The panorama now renders immediately and correctly

### Image Caching Fix
- **Fixed Stale Image Display** - The viewer was showing cached images from previous runs
  - Root cause: Python node reuses the same temp filename (`star_pano_temp_0.jpg`) every run
  - Solution: Added cache-busting timestamps (`&t=<timestamp>`) to all `/view` URLs
  - Each execution now loads the fresh panorama image

## 📚 Documentation

- Updated `web/docs/StarPanoramaViewer.md` with comprehensive control bar documentation
- Added detailed descriptions of all interactive controls and features
- Updated limitations section to reflect fullscreen availability
- Clarified that the viewer size is fixed at 512×512 in normal mode, with fullscreen available via the control bar

## 📊 Statistics

- **Total Active Nodes:** 89 (unchanged from v2.1.0)
- **Files Changed:** 3 files updated
  - `web/js/star_panorama_viewer.js` - Added control bar and bug fixes
  - `web/docs/StarPanoramaViewer.md` - Updated documentation
  - `README.md`, `CHANGELOG.md`, `pyproject.toml` - Version bump to 2.1.1

## 🔄 Compatibility

- **No Breaking Changes** - Fully compatible with v2.1.0 workflows
- Existing panorama viewer nodes will automatically gain the new controls
- No changes required to existing workflows

## 📦 Files Included in Release

This release folder contains only the files that were updated for v2.1.1:

```
release2.1.1/
├── README.md                              # Updated version number and release notes
├── CHANGELOG.md                           # Added v2.1.1 entry
├── pyproject.toml                         # Version bump to 2.1.1
├── RELEASE_NOTES_2.1.1.md                # This file
├── image_tools/
│   └── star_panorama_viewer.py           # Original node (unchanged, included for reference)
└── web/
    ├── js/
    │   └── star_panorama_viewer.js       # Enhanced with control bar and bug fixes
    └── docs/
        └── StarPanoramaViewer.md         # Updated documentation
```

## 🚀 Upgrade Instructions

### Via Git
```bash
cd ComfyUI/custom_nodes/comfyui_starnodes
git pull
```

### Manual Update
Replace the following files in your installation:
1. `web/js/star_panorama_viewer.js`
2. `web/docs/StarPanoramaViewer.md`
3. `README.md`
4. `CHANGELOG.md`
5. `pyproject.toml`

Then restart ComfyUI and refresh your browser (Ctrl+R or Cmd+R).

## 💡 Usage Tips

1. **Auto-Rotation for Previews** - Enable auto-rotation with the play button to get a quick 360° preview of your panorama
2. **Speed Control** - Use negative speed values to reverse the rotation direction
3. **Fullscreen for Detail** - Use fullscreen mode to inspect panorama quality and check for seams
4. **Hold to Pan** - The arrow buttons respond to press-and-hold for continuous smooth panning
5. **Reset Often** - Use the reset button to quickly return to the default view when you get disoriented

## 🔮 Future Plans

- Depth map integration for enhanced parallax effects
- Customizable viewer size options
- VR headset support
- Gyroscope controls for mobile devices

---

**Thank you for using ComfyUI StarNodes!**

For issues, feature requests, or contributions, please visit:
https://github.com/Starnodes2024/ComfyUI_StarNodes
