# Star 360 Parallax Viewer

## Description
An interactive 360-degree panorama viewer with real-time parallax effects for stereoscopic images. This node displays equirectangular panoramas in an embedded Three.js viewer within ComfyUI, supporting mono, side-by-side (SBS), and top/bottom stereoscopic formats with mouse-controlled navigation and parallax displacement.

## Inputs

### Required
- **image**: The panorama image to display (IMAGE type)
  - Supports equirectangular 360° panoramas
  - Can be mono or stereoscopic (SBS/Top-Bottom)
- **layout**: Format of the input image
  - **Mono**: Standard single-view panorama
  - **SBS**: Side-by-Side stereoscopic (left/right eyes horizontally)
  - **Top/Bottom**: Over-under stereoscopic (left/right eyes vertically) - Default

### Optional
- **depth_map**: Depth map for enhanced parallax effects (IMAGE type)
  - Currently cached but not yet fully integrated into parallax calculations
  - Reserved for future depth-based displacement enhancements

## Outputs
This is an output-only node with no return values. It displays the panorama in an interactive viewer widget embedded in the node.

## Features

### Interactive 360° Viewing
- **Full Sphere Navigation**: View the entire 360° panorama by dragging with the mouse
- **Smooth Camera Movement**: Interpolated rotation for fluid viewing experience
- **Field of View Control**: Scroll to zoom in/out (30°-120° FOV range)
- **On-Screen Control Bar**: Pan arrows, reset, zoom, auto-rotation, and fullscreen buttons at the bottom of the viewer

### Auto-Rotation
- **Play/Pause Button (▶/⏸)**: Start or stop automatic horizontal rotation of the panorama
- **Speed Slider**: Adjust rotation speed from -5 to +5 (negative values reverse the direction, 0 pauses)

### Stereoscopic Parallax
When using **SBS** or **Top/Bottom** layouts:
- **Mouse-Based Parallax**: Moving the mouse creates a subtle 3D parallax effect
- **Dynamic UV Offset**: Real-time texture coordinate adjustment based on mouse position
- **Depth Simulation**: Creates the illusion of depth on a 2D monitor without requiring VR headsets

### Technical Implementation
- **Three.js Integration**: Automatically loads Three.js from CDN if not present
- **WebGL Rendering**: Hardware-accelerated 3D graphics
- **Inside-Out Sphere**: Inverted sphere geometry for correct panorama viewing
- **Texture Mapping**: Proper UV coordinate handling for different stereoscopic formats

## Usage

### Basic Mono Panorama
1. Connect a 360° equirectangular image to the **image** input
2. Set **layout** to "Mono"
3. Execute the workflow
4. **Interact with the viewer**:
   - **Drag** to look around
   - **Scroll** to zoom in/out

### Stereoscopic Panorama (Top/Bottom)
1. Connect a stereoscopic panorama (top half = left eye, bottom half = right eye) to **image**
2. Set **layout** to "Top/Bottom"
3. Execute the workflow
4. **Move your mouse** over the viewer to see the parallax effect
5. **Drag** to navigate the 360° scene

### Stereoscopic Panorama (SBS)
1. Connect a side-by-side stereoscopic panorama (left half = left eye, right half = right eye) to **image**
2. Set **layout** to "SBS"
3. Execute the workflow
4. **Move your mouse** for parallax displacement
5. **Drag** to explore the panorama

## Viewer Controls

### Mouse Controls
- **Left Click + Drag**: Rotate camera to look around the 360° scene
- **Mouse Move** (SBS/Top-Bottom only): Create parallax displacement effect
- **Scroll Wheel**: Adjust field of view (zoom in/out)

### Control Bar
A control bar is shown at the bottom of the viewer once a panorama is loaded:
- **◀ ▲ ▼ ▶**: Pan the view in the given direction (press and hold for continuous movement)
- **⌂ (Reset)**: Return the camera to the initial orientation and zoom level
- **− / +**: Zoom out / zoom in (same range as the scroll wheel)
- **▶ / ⏸**: Toggle auto-rotation of the panorama
- **Speed Slider**: Set auto-rotation speed and direction (-5 to +5, negative = reverse)
- **⛶ (Fullscreen)**: Toggle fullscreen viewing; press again or hit Esc to exit

### Camera Behavior
- **Smooth Interpolation**: Camera rotation smoothly follows your drag movements
- **Vertical Limit**: Camera pitch is clamped to ±90° to prevent disorientation
- **Continuous Rotation**: Horizontal rotation is unlimited for full 360° exploration

## Technical Details

### Texture Mapping
- **Mono**: Full texture mapped to sphere (1:1 UV coordinates)
- **Top/Bottom**: Top half mapped by default, bottom half used for parallax
  - UV repeat: (1, 0.5)
  - UV offset: (0, 0.5) base, adjusted by mouse
- **SBS**: Left half mapped by default, right half used for parallax
  - UV repeat: (0.5, 1)
  - UV offset: (0, 0) base, adjusted by mouse

### Parallax Algorithm
For stereoscopic layouts, the viewer:
1. Tracks mouse position relative to canvas center
2. Calculates normalized coordinates (-1 to 1)
3. Applies offset to texture UV coordinates
4. Blends information from the second eye's view
5. Creates depth perception through subtle displacement

### Performance
- **Canvas Size**: 512×512 pixels (default)
- **Sphere Resolution**: 60×40 segments for smooth rendering
- **Frame Rate**: 60 FPS target with requestAnimationFrame
- **Texture Quality**: Saved at 90% JPEG quality for balance

## Use Cases
- **VR Content Preview**: Preview 360° renders before exporting to VR platforms
- **Stereoscopic Validation**: Check alignment and depth in SBS/TB panoramas
- **Interactive Presentations**: Embed navigable panoramas in workflows
- **Depth Map Visualization**: Combine with depth maps for enhanced effects
- **Panorama Quality Control**: Inspect seams and stitching artifacts interactively

## Workflow Integration
This node works seamlessly with:
- **Star Save Panorama JPG+**: View the 3D output directly
- **Star Image Shifter**: Adjust seam position before viewing
- Any node that outputs equirectangular images

## Notes
- The viewer uses temporary file storage (`/temp` directory)
- Images are automatically cleaned up by ComfyUI's temp management
- Three.js is loaded from CDN on first use (requires internet connection)
- The viewer widget is embedded directly in the node (no external windows)
- Parallax effect strength is calibrated for subtle, comfortable viewing
- For best results, use high-resolution equirectangular images (4K+)
- Stereoscopic images should have proper left/right eye alignment

## Limitations
- Depth map input is currently reserved for future enhancements
- Viewer size is fixed at 512×512 (fullscreen mode available via the control bar)
- Requires modern browser with WebGL support
- Three.js CDN dependency (offline use requires manual Three.js installation)

## Example Workflows
1. **Render → View**: Connect a panorama render directly to the viewer for instant preview
2. **Generate → Save → View**: Use Star Save Panorama JPG+ to save and view simultaneously
3. **Depth → 3D → View**: Generate depth map, create stereoscopic image, view with parallax
4. **Batch Preview**: Process multiple panoramas and view each in sequence

This node brings interactive 360° viewing directly into your ComfyUI workspace, eliminating the need for external panorama viewers or VR headsets for quick previews and quality checks.
