# Star PSD Saver (Dynamic)

## Description
This node saves multiple image layers and their associated masks to a Photoshop PSD file. It allows for dynamic input connections, automatically detecting and processing any connected layers and masks. The resulting PSD file maintains layer order and transparency information, making it ideal for post-processing in image editing applications.

## Inputs

### Required
- **filename_prefix**: Base name for the output PSD file (default: "multilayer")
- **output_dir**: Directory where the PSD file will be saved (default: "ComfyUI/output/PSD_Layers")
- **save_psd**: Toggle (default: True). When enabled, writes the layered .psd file to disk. When disabled, no file is written and only the flattened image is output — useful if you only need the composed image for further processing in the workflow.

### Optional
- **layer1**: First image layer (IMAGE type)
- **mask1**: Mask for the first layer (MASK type)
- **Additional dynamic inputs**: The node automatically accepts any number of connected layer/mask pairs

## Outputs
- **flattened_image** (IMAGE): A single flattened composite of all connected layers (respecting masks), useful for feeding into other nodes without needing to open the saved PSD.

The .psd file is additionally saved to disk at the specified location when **save_psd** is enabled.

## Usage
1. Connect one or more image outputs to the layer inputs (layer1, layer2, etc.)
2. Optionally connect corresponding mask outputs to the mask inputs (mask1, mask2, etc.)
3. Set the filename prefix and output directory
4. Toggle **save_psd** on/off depending on whether you want the .psd file written to disk
5. Execute the workflow — the flattened image is always output, and the .psd file is saved if enabled

## Features
- **Dynamic Layer Support**: Automatically handles any number of connected layers
- **Mask Integration**: Properly applies transparency masks to each layer
- **Layer Ordering**: Maintains the order of layers in the PSD file
- **Auto-Centering**: Centers smaller images within the PSD canvas
- **Automatic Filename Management**: Prevents overwriting existing files by adding incremental numbers
- **RGB Conversion**: Ensures all layers are properly converted to RGB mode
- **Flattened IMAGE Output**: Always outputs a flattened composite image, independent of whether the .psd file is saved
- **Optional PSD Saving**: Skip writing to disk entirely via the save_psd toggle, for pure in-workflow compositing

## Notes
- Layers are stacked in numerical order (layer1 at the bottom, layer2 above it, etc.)
- If a layer has no corresponding mask, it will be saved as a fully opaque layer
- The PSD dimensions are determined by the largest connected image
- Smaller images are automatically centered within the canvas
- Masks are properly applied as layer transparency
- The node creates the output directory if it doesn't exist (only relevant when save_psd is enabled)
- The flattened_image output is generated regardless of the save_psd setting

This node is particularly useful for workflows that generate multiple image variations or components that need to be composed in Photoshop for further editing.
