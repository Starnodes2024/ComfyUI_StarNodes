# StarNodes 2.1.0 - Release Notes

**Release Date:** January 2025

## What's New

### New Nodes (2)

#### Image Manipulation & Helpers
- **⭐ Star Box Drawer** - Draw rectangular boxes on images with customizable colors, positions, and sizes
  - Supports both filled and outlined rectangles
  - 5 color options: white, red, blue, green, black
  - Adjustable line width for outlines
  - Perfect for masking, highlighting, and visual debugging

- **⭐ Star Image Shifter** - Shift images with seamless wrapping
  - Horizontal and vertical shifting with wrap-around
  - Ideal for panoramas and tileable textures
  - Adjust seam positions in 360° images
  - Supports shifts from -8192 to +8192 pixels

### Enhancements

#### Star Save Panorama JPG+
- **New Output:** Added `3d_image` output connector
  - Outputs the stereoscopic 3D image (SBS or Top/Bottom) when enabled
  - Allows further processing of the 3D image in the workflow
  - Returns blank placeholder when stereo_3d is disabled

### Documentation
- Added comprehensive help files for all new nodes in `web/docs`:
  - `StarBoxDrawer.md`
  - `StarImageShifter.md`
  - Updated `StarSavePanoramaJPEGPlus.md`

## Total Node Count
**88 active nodes** (86 from v2.0.1 + 2 new helper nodes)

## Installation

### New Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes.git
cd ComfyUI_StarNodes
pip install -r requirements.txt
```

### Update from Previous Version
```bash
cd ComfyUI/custom_nodes/ComfyUI_StarNodes
git pull
pip install -r requirements.txt --upgrade
```

## Compatibility
- ComfyUI: Latest version recommended
- Python: 3.10+
- No breaking changes from v2.0.1

## Known Issues
None reported for this release.

## What's Next
Stay tuned for more helper nodes and workflow enhancements in future releases!

---

**Full Changelog:** See `CHANGELOG.md` for detailed changes.
