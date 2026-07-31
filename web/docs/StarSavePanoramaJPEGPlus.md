# Star Save Panorama JPG+

## Description
An enhanced version of the Star Save Panorama JPEG node that saves images as JPEG files with panorama XMP metadata for 360° viewers. This node adds advanced features including flexible path management, optional stereoscopic 3D output generation from depth maps, and an IMAGE output connector for workflow integration.

## Inputs

### Required
- **images**: The images to save (IMAGE type)
- **subfolder**: Optional subfolder inside the base path (default: "")
- **filename**: Base filename for the saved images (default: "panorama")
- **quality**: JPEG quality from 1-100 (default: 95)
- **stereo_3d**: Enable to generate and save a stereoscopic 3D version (default: False)
- **stereo_layout**: Layout for the 3D image — "SBS" (Side-by-Side) or "Top/Bottom" (default: "SBS")
- **depth_scale**: Strength of the 3D effect. Higher values shift pixels further (default: 5.0, range: 0.1-100.0)
- **invert_depth**: Enable if the depth map uses white for far instead of near (default: False)
- **gap_fill**: Method to fill gaps created by pixel shifting — "Inpaint" (best quality) or "None" (default: "Inpaint")

### Optional
- **base_path**: Base save folder path (e.g., from Star Save Image+ path output). If not connected, uses ComfyUI's default output directory
- **depth_map**: Depth map used to generate the stereoscopic image (required when stereo_3d is enabled)

## Outputs
- **image** (IMAGE): The original input images, passed through for use in other nodes within the same workflow
- **3d_image** (IMAGE): The stereoscopic 3D image (SBS or Top/Bottom format) if stereo_3d is enabled. Returns a blank placeholder if stereo_3d is disabled or no depth_map is provided.

The JPEG files are saved to disk at the specified location with panorama XMP metadata injected.

## Usage

### Basic Panorama Saving
1. Connect image outputs to the **images** input
2. Set **filename** and **subfolder** as desired
3. Adjust **quality** if needed (95 is recommended for panoramas)
4. Execute the workflow — the images are saved with panorama metadata and also output for further processing

### Stereoscopic 3D Panorama
1. Connect your panorama image to **images**
2. Connect a depth map to **depth_map** (same dimensions as the panorama)
3. Enable **stereo_3d**
4. Choose **stereo_layout** (SBS for side-by-side, Top/Bottom for over-under)
5. Adjust **depth_scale** to control the 3D effect strength (5.0 is a good starting point)
6. Enable **invert_depth** if your depth map is inverted (white = far)
7. Execute — both the original panorama and the stereoscopic version are saved

### Path Management
- **Without base_path**: Saves to `ComfyUI/output/[subfolder]/[filename]_00001_.jpg`
- **With base_path**: Saves to `[base_path]/[subfolder]/[filename]_00001_.jpg`
- Use the **path** output from Star Save Image+ as **base_path** to keep all outputs in the same folder

## Features
- **Panorama XMP Metadata**: Automatically injects Google Photo Sphere XMP metadata for 360° viewer compatibility
- **Stereoscopic 3D Generation**: Creates side-by-side or top/bottom 3D images from depth maps using pixel shifting
- **Intelligent Gap Filling**: Uses OpenCV inpainting to fill gaps created by depth-based pixel shifting
- **Flexible Path Management**: Works with or without base_path input for maximum workflow flexibility
- **IMAGE Output**: Pass-through output allows using the saved images in subsequent nodes
- **Batch Processing**: Handles multiple images in a single execution
- **Automatic Filename Management**: Prevents overwriting with incremental counters

## Stereoscopic 3D Technical Details
The stereo generation algorithm:
1. Converts the depth map to grayscale if needed
2. Normalizes depth values to pixel shift amounts based on **depth_scale**
3. Shifts each pixel horizontally (for SBS) based on its depth value
4. Fills gaps created by shifting using inpainting (if enabled)
5. Combines the original and shifted images side-by-side or top/bottom
6. Injects panorama XMP metadata into both the original and stereoscopic files

The resulting stereoscopic image can be viewed with VR headsets or 3D-capable panorama viewers.

## Notes
- The panorama XMP metadata marks images as cylindrical 360° panoramas for viewers like Google Photos, Facebook 360, and VR platforms
- Stereoscopic files are saved with `_SBS.jpg` or `_TB.jpg` suffix
- Both the original panorama and stereoscopic version receive panorama XMP metadata
- Depth maps should match the panorama dimensions (automatic resizing is applied if needed)
- Higher **depth_scale** values create stronger 3D effects but may introduce more visible gaps
- Inpainting gap fill provides the best quality but adds processing time
- The **image** output is the original input, not the saved file — use this to continue processing in the workflow
- The **3d_image** output contains the generated stereoscopic image for further workflow processing (e.g., additional filters, preview, or re-saving)

## Example Workflows
- Save equirectangular renders as panoramas for VR viewing
- Generate stereoscopic 360° images from panorama + depth map pairs
- Batch-process multiple panoramas with consistent settings
- Integrate panorama saving into larger workflows using the IMAGE output

This node is ideal for VR content creation, 360° photography workflows, and stereoscopic panorama generation.
