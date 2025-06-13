# Starnodes Aspect Video Ratio

## Description
This node allows you to select a standard video aspect ratio from a dropdown and input a desired width. It automatically calculates the corresponding height and outputs width and height as both integers and strings, as well as a formatted size string (e.g., "750x422").

## Inputs
- **ratio**: Select from common video and photo aspect ratios (e.g., "16:9 [landscape]", "4:3 [landscape]", "9:16 [portrait]", etc.).
- **width**: The desired width in pixels (default: 750).

## Outputs
- **width**: The input width as an integer.
- **width_str**: The width as a string.
- **height**: The calculated height as an integer, based on the selected ratio.
- **height_str**: The height as a string.
- **size**: The formatted size string in the form "widthxheight" (e.g., "750x422").

## Usage
1. Select the desired aspect ratio from the dropdown menu.
2. Enter the width (in pixels) or use the default value.
3. The node will calculate the height for you and provide all outputs for use in downstream nodes.

### Supported Ratios
- Free Ratio (width = height)
- 1:1 [square]
- 8:5 [landscape]
- 4:3 [landscape]
- 3:2 [landscape]
- 7:5 [landscape]
- 16:9 [landscape]
- 21:9 [landscape]
- 19:9 [landscape]
- 3:4 [portrait]
- 2:3 [portrait]
- 5:7 [portrait]
- 9:16 [portrait]
- 9:21 [portrait]
- 5:8 [portrait]
- 9:19 [portrait]

## Example
- **Input**: ratio = "16:9 [landscape]", width = 750
- **Output**: width = 750, width_str = "750", height = 422, height_str = "422", size = "750x422"

This node is ideal for generating video or image dimensions that match common aspect ratios, simplifying workflow setup for video and image processing in ComfyUI.
