# Star 360 Parallax Viewer Pro

## Description
An enhanced version of the Star 360 Parallax Viewer that exports an image batch for video creation. It displays the same interactive 360-degree panorama viewer with parallax effects, plus adds a green frame overlay showing the export dimensions, and renders all frames as an IMAGE batch output.

## Inputs

### Required
- **image**: The panorama image to display (IMAGE type)
  - Supports equirectangular 360° panoramas (mono or stereoscopic)
- **layout**: Format of the input image
  - **Mono**: Standard single-view panorama
  - **SBS**: Side-by-Side stereoscopic
  - **Top/Bottom**: Over-under stereoscopic (Default)
- **resolution**: Output resolution preset
  - **HD (1280x720)**: Base width 1280
  - **Full HD (1920x1080)**: Base width 1920 (Default)
- **ratio**: Aspect ratio of the output video frame
  - Same ratios as Star Ratios Latent Advanced: 1:1, 1:2, 3:4, 2:3, 5:7, 9:16, 9:21, 10:16, 4:3, 16:10, 3:2, 2:1, 7:5, 16:9, 21:9, custom
  - Default: 16:9
  - When a ratio is selected, a green frame box appears on the viewer showing the exact export dimensions
- **custom_ratio**: Custom aspect ratio string (used when ratio is set to "custom")
  - Format: "W:H" (e.g. "21:9") or "WxH"
- **framerate**: Frames per second preset
  - Options: 24, 25, 30, 50, 60
  - Default: 30
- **direction**: Rotation direction
  - **Right to Left**: Camera pans right to left (Default)
  - **Left to Right**: Camera pans left to right
- **num_loops**: Number of complete 360° rotations
  - One loop = frames for one whole 360° move minus 1 (last frame would be identical to first)
  - Default: 1, Range: 1-100
- **zoom**: Zoom level (affects field of view)
  - 1.0 = default 75° FOV, higher values zoom in (narrower FOV)
  - Default: 1.0, Range: 0.5-5.0
- **speed**: Rotation speed in degrees per second
  - Default: 30.0, Range: 1.0-360.0
  - Higher values = faster rotation = fewer frames per loop

### Optional
- **depth_map**: Depth map for enhanced parallax effects (IMAGE type)

## Outputs
- **frames**: IMAGE batch containing all rendered frames for video creation
  - Shape: [total_frames, height, width, 3]
  - Can be connected to video encoder nodes or SaveImage nodes

## Features

### Interactive Viewer (same as original)
- Full 360° navigation with mouse drag
- Scroll wheel zoom (30°-120° FOV)
- Auto-rotation with speed slider
- Parallax effect for stereoscopic layouts
- Fullscreen mode
- Control bar with pan arrows, reset, zoom, play/pause

### Green Frame Overlay
- Shows a green rectangle on the viewer representing the export aspect ratio
- Updates in real-time when ratio, resolution, or custom_ratio widgets change
- Displays the exact export dimensions (e.g. "1920x1080") as a label

### Frame Export
- Renders perspective views from the equirectangular panorama
- Each frame represents one step of the 360° rotation
- Total frames = frames_per_loop × num_loops
  - frames_per_loop = (360 / (speed / framerate)) - 1
- For stereoscopic layouts, renders from the left eye view
- Maximum 5000 frames per execution

## Usage

1. Connect a 360° equirectangular image to the **image** input
2. Set **layout** to match your image format
3. Choose **resolution** (HD or Full HD)
4. Select **ratio** for the output video aspect ratio
5. Set **framerate**, **direction**, **num_loops**, **zoom**, and **speed**
6. Execute the workflow
7. The **frames** output contains all rendered frames as an image batch
8. Connect to a video encoder (e.g. Video Combine) or SaveImage node

## Frame Count Calculation

Example: speed=30°/s, framerate=30fps, num_loops=1
- Degrees per frame = 30/30 = 1.0°
- Frames per loop = 360/1.0 - 1 = 359
- Total frames = 359 × 1 = 359

Example: speed=60°/s, framerate=30fps, num_loops=2
- Degrees per frame = 60/30 = 2.0°
- Frames per loop = 360/2.0 - 1 = 179
- Total frames = 179 × 2 = 358

## Notes
- The green frame overlay is for visual reference only; it shows the aspect ratio of the exported frames
- For stereoscopic inputs, frames are rendered from the left eye perspective
- Large frame counts at high resolution may use significant memory
- The viewer uses Three.js for interactive preview; frame export uses numpy-based equirectangular-to-perspective projection
