# Star Grid Captions Batcher

## Description
The Star Grid Captions Batcher is a companion node to the Star Grid Composer that collects multiple caption strings and combines them into a single formatted batch. This node makes it easy to prepare captions from different sources to be displayed alongside their corresponding images in a grid layout.

## Inputs

### Optional
- **caption 1**: First caption string to include in the grid
- (Additional caption inputs are dynamically available in the UI, up to 100 captions)

## Outputs
- **Grid Captions Batch**: A formatted string containing all captions, ready for use with the Star Grid Composer

## Usage
1. Connect text strings to the "caption 1", "caption 2", etc. inputs
2. Connect the "Grid Captions Batch" output to the Star Grid Composer's "Grid Captions Batch" input
3. The captions will be displayed with their corresponding images in the grid

## Features

### Flexible Caption Collection
- Accepts up to 100 individual caption inputs
- Maintains the order of captions to match their corresponding images
- Handles empty captions gracefully

### Automatic Formatting
- Combines all captions into a properly formatted batch
- Removes trailing empty captions to optimize processing
- Preserves caption order for proper alignment with images

### Seamless Integration
- Works directly with the Star Grid Composer
- Captions are automatically aligned with their corresponding images in the grid
- Supports any text content, including multi-line captions

## Technical Details
- The node collects all non-empty caption strings from the inputs
- Empty captions are included to maintain position alignment with images
- Trailing empty captions are removed for efficiency
- The final output is a newline-separated string containing all captions

## Notes
- For best results, provide captions in the same order as the images in the Star Grid Image Batcher
- Empty captions will result in images with no caption text
- The Star Grid Composer must have "add_caption" set to true to display the captions
- Caption styling (font, size, colors) is controlled by the Star Grid Composer node
- This node works seamlessly with the Star Grid Composer and Star Grid Image Batcher
