# Star Size Calculator by Side

## Overview

The **Star Size Calculator by Side** node is a utility tool that calculates new image dimensions based on resizing either the longest or shortest side while maintaining the original aspect ratio.

## Category

`⭐StarNodes/Helpers And Tools`

## Inputs

### Optional Inputs

- **image** (IMAGE): An input image to use as the source for dimensions
- **width** (INT): Manual width input (default: 1024, range: 1-16384)
- **height** (INT): Manual height input (default: 1024, range: 1-16384)

### Required Inputs

- **use_input_image** (BOOLEAN): Toggle to use the input image dimensions or manual width/height inputs (default: True)
- **target_size** (INT): The target size in pixels for the selected side (default: 1024, range: 1-16384)
- **resize_by_side** (SELECTOR): Choose which side to resize by:
  - **Longest**: Resize based on the longest dimension
  - **Shortest**: Resize based on the shortest dimension

## Outputs

- **width_str** (STRING): The calculated width as a string
- **width_int** (INT): The calculated width as an integer
- **height_str** (STRING): The calculated height as a string
- **height_int** (INT): The calculated height as an integer
- **long_side_str** (STRING): The longest dimension as a string
- **long_side_int** (INT): The longest dimension as an integer
- **short_side_str** (STRING): The shortest dimension as a string
- **short_side_int** (INT): The shortest dimension as an integer

## How It Works

The node calculates new dimensions while preserving the aspect ratio:

1. **Source Dimensions**: Either from the input image (if `use_input_image` is True) or from manual width/height inputs
2. **Aspect Ratio Calculation**: Determines the ratio between width and height
3. **Side Selection**: 
   - If **Longest** is selected, the longer dimension is set to `target_size`
   - If **Shortest** is selected, the shorter dimension is set to `target_size`
4. **Dimension Calculation**: The other dimension is calculated to maintain the aspect ratio

## Use Cases

### Example 1: Resize by Longest Side
- Input image: 1920x1080 (landscape)
- Target size: 1024
- Resize by: Longest
- Result: 
  - Width: 1024, Height: 576
  - Long side: 1024, Short side: 576

### Example 2: Resize by Shortest Side
- Input image: 1920x1080 (landscape)
- Target size: 512
- Resize by: Shortest
- Result: 
  - Width: 910, Height: 512
  - Long side: 910, Short side: 512

### Example 3: Manual Dimensions
- Use input image: False
- Width: 2048
- Height: 1536
- Target size: 768
- Resize by: Longest
- Result: 
  - Width: 768, Height: 576
  - Long side: 768, Short side: 576

## Workflow Integration

This node is particularly useful for:
- Preparing images for specific resolution requirements
- Batch processing images to consistent sizes
- Creating thumbnails or previews
- Ensuring images fit within maximum dimension constraints
- Preprocessing images before upscaling or model inference

## Tips

- Use **Longest** when you want to ensure images don't exceed a maximum dimension
- Use **Shortest** when you want to ensure a minimum dimension is met
- The aspect ratio is always preserved, so images won't be distorted
- Both string and integer outputs are provided for compatibility with different nodes
- Use the **long_side** and **short_side** outputs when you need to work with dimensions without checking the orientation - these automatically give you the longest and shortest dimensions regardless of whether the image is landscape or portrait
