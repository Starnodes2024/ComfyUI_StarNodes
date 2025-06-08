# Star PSD Saver (Dynamic)

## Description
This node saves multiple image layers and their associated masks to a Photoshop PSD file. It allows for dynamic input connections, automatically detecting and processing any connected layers and masks. The resulting PSD file maintains layer order and transparency information, making it ideal for post-processing in image editing applications.

## Inputs

### Required
- **filename_prefix**: Base name for the output PSD file (default: "multilayer")
- **output_dir**: Directory where the PSD file will be saved (default: "ComfyUI/output/PSD_Layers")

### Optional
- **layer1**: First image layer (IMAGE type)
- **mask1**: Mask for the first layer (MASK type)
- **Additional dynamic inputs**: The node automatically accepts any number of connected layer/mask pairs

## Outputs
This is an output node with no return values. The PSD file is saved to disk at the specified location.

## Usage
1. Connect one or more image outputs to the layer inputs (layer1, layer2, etc.)
2. Optionally connect corresponding mask outputs to the mask inputs (mask1, mask2, etc.)
3. Set the filename prefix and output directory
4. Execute the workflow to save the PSD file

## Features
- **Dynamic Layer Support**: Automatically handles any number of connected layers
- **Mask Integration**: Properly applies transparency masks to each layer
- **Layer Ordering**: Maintains the order of layers in the PSD file
- **Auto-Centering**: Centers smaller images within the PSD canvas
- **Automatic Filename Management**: Prevents overwriting existing files by adding incremental numbers
- **RGB Conversion**: Ensures all layers are properly converted to RGB mode

## Notes
- Layers are stacked in numerical order (layer1 at the bottom, layer2 above it, etc.)
- If a layer has no corresponding mask, it will be saved as a fully opaque layer
- The PSD dimensions are determined by the largest connected image
- Smaller images are automatically centered within the canvas
- Masks are properly applied as layer transparency
- The node creates the output directory if it doesn't exist

This node is particularly useful for workflows that generate multiple image variations or components that need to be composed in Photoshop for further editing.
