# Star Image Shifter

## Description
A utility node that shifts (wraps) images horizontally and vertically with seamless wrapping. The image content wraps around the edges, making it ideal for panoramas, tileable textures, and seamless pattern adjustments. Supports both positive and negative shifts in both axes.

## Inputs

### Required
- **image**: The input image to shift (IMAGE type)
- **x_shift**: Horizontal shift in pixels (default: 0, range: -8192 to 8192)
  - Positive values shift the image to the right
  - Negative values shift the image to the left
- **y_shift**: Vertical shift in pixels (default: 0, range: -8192 to 8192)
  - Positive values shift the image down
  - Negative values shift the image up

## Outputs
- **IMAGE**: The shifted image with wrapped edges

## Usage

### Horizontal Shift
1. Connect an image to the **image** input
2. Set **x_shift** to a non-zero value (e.g., 100 to shift right, -100 to shift left)
3. Leave **y_shift** at 0
4. Execute — the image is shifted horizontally with content wrapping from the opposite edge

### Vertical Shift
1. Connect an image to the **image** input
2. Leave **x_shift** at 0
3. Set **y_shift** to a non-zero value (e.g., 50 to shift down, -50 to shift up)
4. Execute — the image is shifted vertically with content wrapping from the opposite edge

### Combined Shift
1. Connect an image to the **image** input
2. Set both **x_shift** and **y_shift** to non-zero values
3. Execute — the image is shifted in both directions with seamless wrapping

## Features
- **Seamless Wrapping**: Content that moves off one edge reappears on the opposite edge
- **Bidirectional Shifting**: Shift in both horizontal and vertical directions simultaneously
- **Modulo Wrapping**: Shift values automatically wrap using modulo arithmetic, so shifts larger than the image dimensions are handled correctly
- **Batch Support**: Processes all images in a batch with the same shift values
- **Panorama-Friendly**: Ideal for adjusting the seam position in 360° panoramas

## Technical Details
The shifting algorithm:
1. Applies modulo wrapping to the shift values based on image dimensions
2. Splits the image into sections at the wrap points
3. Rearranges the sections to create the shifted result
4. First applies horizontal shift, then vertical shift

For example, with `x_shift=100` on a 1000px wide image:
- Pixels 900-1000 move to positions 0-100
- Pixels 0-900 move to positions 100-1000

## Use Cases
- **Panorama Seam Adjustment**: Move the seam of a 360° panorama to a less visible location
- **Tileable Texture Alignment**: Adjust the starting point of repeating patterns
- **Image Registration**: Align images that are offset from each other
- **Seamless Pattern Creation**: Fine-tune the wrapping point of seamless textures
- **Equirectangular Image Editing**: Reposition the center point of equirectangular projections
- **Testing Seamlessness**: Verify that an image tiles correctly by shifting and checking for visible seams

## Notes
- Shift values are automatically wrapped using modulo, so `x_shift=1100` on a 1000px wide image is equivalent to `x_shift=100`
- The wrapping is seamless only if the image content is designed to tile (e.g., panoramas, seamless textures)
- For non-seamless images, visible seams will appear at the wrap boundaries
- The same shift is applied to all images in a batch
- Shift values of 0 return the original image unchanged
- Negative shifts are supported and wrap in the opposite direction

## Example Workflows
- Shift a 360° panorama to move the seam to the back of the scene
- Adjust a tileable texture so the pattern starts at a different position
- Align two panoramas that were captured with different camera orientations
- Test if a texture is truly seamless by shifting and checking for discontinuities

This node is particularly useful for panoramic image workflows and seamless texture creation in ComfyUI.
