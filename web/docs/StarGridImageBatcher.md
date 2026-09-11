# Star Grid Image Batcher

## Description
The Star Grid Image Batcher is a utility node designed to work with the Star Grid Composer. It collects multiple individual images or image batches and combines them into a single batch with consistent dimensions. This node makes it easy to prepare images from different sources for grid composition, ensuring they all have compatible sizes and formats.

## Inputs

### Optional
- **image_batch**: A batch of images to include in the grid
- **image 1**: First individual image to include in the grid
- (Additional image inputs are dynamically available in the UI, up to 100 images)

## Outputs
- **Grid Image Batch**: A batch of processed images ready for use with the Star Grid Composer

## Usage
1. Connect individual images to the "image 1", "image 2", etc. inputs
2. And/or connect a batch of images to the "image_batch" input
3. Connect the "Grid Image Batch" output to the Star Grid Composer's "Grid Image Batch" input
4. The node will process all images to have consistent dimensions

## Features

### Flexible Input Handling
- Accepts both individual images and batches of images
- Dynamically supports up to 100 individual image inputs
- Automatically detects and processes different input formats

### Automatic Image Processing
- Resizes all images to have the same dimensions
- Preserves aspect ratios during resizing
- Centers images within a square canvas
- Pads images with black borders as needed

### Batch Optimization
- Efficiently combines multiple image sources into a single tensor
- Handles different image shapes and formats
- Ensures all images are compatible with the Star Grid Composer

## Technical Details
- The node first collects all valid images from both batch and individual inputs
- It determines the maximum dimension across all images
- Each image is then resized proportionally to fit within this maximum dimension
- Images are padded to create square outputs with the content centered
- The result is a batch tensor with shape [N, C, H, W] where:
  - N is the number of images
  - C is the number of channels (3 for RGB)
  - H and W are equal (square) and match the maximum dimension

## Notes
- For best results, connect images in the order you want them to appear in the grid
- Empty or missing inputs are skipped
- If no valid images are provided, the node returns an empty batch
- The node works seamlessly with the Star Grid Composer and Star Grid Captions Batcher
- When using with the Star Grid Composer, ensure the grid dimensions (rows × columns) can accommodate all your images
