# Star Grid Composer

## Description
The Star Grid Composer is a versatile node that arranges multiple images into a customizable grid layout. It automatically handles image sizing, aspect ratio preservation, and can add optional captions to both individual images and the entire composition. This node is particularly useful for creating comparison grids, batch result displays, and presentation-ready image collections.

## Inputs

### Required
- **max_width**: Maximum width of the output grid in pixels
- **max_height**: Maximum height of the output grid in pixels
- **cols**: Number of columns in the grid
- **rows**: Number of rows in the grid
- **Grid Image Batch**: Batch of images to arrange in the grid (can be a single image or multiple images)
- **add_caption**: Whether to add captions to images
- **caption_font_size**: Font size for individual image captions
- **main_caption_font_size**: Font size for the main caption
- **caption_bg_color**: Background color for individual image captions
- **caption_text_color**: Text color for individual image captions
- **font_family**: Font family to use for all captions
- **font_bold**: Whether to use bold font for captions
- **main_caption_bg_color**: Background color for the main caption
- **main_caption_text_color**: Text color for the main caption

### Optional
- **input_caption**: Text for the main caption (displayed at the bottom of the grid)
- **Grid Captions Batch**: Batch of captions for individual images

## Outputs
- **image**: The composed grid image

## Usage
1. Connect a batch of images to the "Grid Image Batch" input
2. Set the desired grid dimensions (columns and rows)
3. Optionally enable captions and connect caption text
4. Adjust styling parameters as needed
5. The node will automatically arrange the images in a grid with consistent sizing

## Features

### Automatic Image Sizing
- Preserves aspect ratio of all images
- Centers images within their grid cells
- Automatically scales images to fit within the grid dimensions

### Flexible Caption System
- Individual captions for each image
- Optional main caption for the entire grid
- Customizable font, size, and colors
- Support for transparent caption backgrounds

### Dynamic Font Selection
- Automatically detects and lists available system fonts
- Provides fallback fonts if system fonts can't be detected
- Support for bold text styling

### Grid Layout Options
- Customizable number of columns and rows
- Adjustable maximum dimensions
- Automatic padding and spacing

## Technical Details
- Images are arranged in row-major order (left to right, then top to bottom)
- Each image maintains its original aspect ratio within its grid cell
- If fewer images are provided than grid cells, empty cells will be black
- Caption height is automatically calculated based on font size
- The main caption appears at the bottom of the grid

## Notes
- For best results, provide a number of images equal to or less than rows × columns
- When using batch inputs from multiple sources, ensure all batches have the same number of images
- The "transparent" option for caption backgrounds allows for borderless captions
- For high-resolution grids, increase both max_width and max_height proportionally
- Font availability depends on the system where ComfyUI is running
