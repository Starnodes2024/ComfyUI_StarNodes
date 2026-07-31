# Star Box Drawer

## Description
A simple utility node that draws rectangular boxes on images. Supports both filled and outlined rectangles with customizable colors, positions, and sizes. Useful for masking, highlighting regions, creating overlays, or testing image processing workflows.

## Inputs

### Required
- **image**: The input image to draw on (IMAGE type)
- **x**: X-coordinate of the top-left corner of the box (default: 0, range: 0-8192)
- **y**: Y-coordinate of the top-left corner of the box (default: 0, range: 0-8192)
- **width**: Width of the box in pixels (default: 100, range: 1-8192)
- **height**: Height of the box in pixels (default: 100, range: 1-8192)
- **color**: Color of the box — "white", "red", "blue", "green", or "black" (default: "white")
- **filled**: Whether to fill the box or draw only the outline (default: True)
- **line_width**: Width of the outline when filled is False (default: 2, range: 1-50)

## Outputs
- **IMAGE**: The input image with the box drawn on it

## Usage

### Drawing a Filled Box
1. Connect an image to the **image** input
2. Set **x** and **y** to position the top-left corner
3. Set **width** and **height** to define the box size
4. Choose a **color**
5. Keep **filled** enabled (True)
6. Execute — the image is returned with a filled rectangle

### Drawing an Outlined Box
1. Connect an image to the **image** input
2. Set **x**, **y**, **width**, and **height** as desired
3. Choose a **color**
4. Disable **filled** (set to False)
5. Adjust **line_width** to control the outline thickness
6. Execute — the image is returned with an outlined rectangle

## Features
- **Simple Interface**: Straightforward parameters for quick box drawing
- **Filled or Outlined**: Toggle between solid fills and outlines
- **Customizable Colors**: Five common colors available
- **Adjustable Line Width**: Control outline thickness for outlined boxes
- **Batch Support**: Processes all images in a batch, drawing the same box on each
- **Precise Positioning**: Pixel-perfect control over box placement and size

## Use Cases
- **Region Highlighting**: Mark areas of interest in images
- **Masking Preparation**: Create simple rectangular masks
- **Overlay Creation**: Add colored boxes for compositing
- **Testing and Debugging**: Visualize specific regions during workflow development
- **Watermarking**: Add colored boxes as simple watermarks or borders
- **Image Annotation**: Mark regions for further processing

## Notes
- The box is drawn on all images in a batch with the same position and size
- Coordinates are in pixels, with (0, 0) at the top-left corner
- If the box extends beyond the image boundaries, only the visible portion is drawn
- The **line_width** parameter only affects outlined boxes (when **filled** is False)
- For filled boxes, the **line_width** parameter is ignored
- The node modifies the input image directly — use a copy if you need to preserve the original

## Example Workflows
- Draw a red outline box around a detected object region
- Create a white filled box to mask out unwanted areas
- Add colored boxes to mark different zones in an image
- Overlay multiple boxes by chaining multiple Box Drawer nodes

This node is ideal for quick image annotation, masking, and visual debugging in ComfyUI workflows.
