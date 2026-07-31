# Star PSD Saver 2 (Optimized)

## Description
An optimized version of the Star PSD Saver node that saves up to 10 image layers and their associated masks to a Photoshop PSD file. This node features improved performance and a fixed number of input slots for better workflow planning. The resulting PSD file maintains layer order and transparency information, making it ideal for post-processing in image editing applications.

## Inputs

### Required
- **filename_prefix**: Base name for the output PSD file (default: "multilayer2")
- **output_dir**: Directory where the PSD file will be saved (default: "ComfyUI/output/PSD_Layers")

### Optional
- **layer1** through **layer10**: Image layers (IMAGE type)
- **mask1** through **mask10**: Masks for corresponding layers (MASK type)

## Outputs
This is an output node with no return values. The PSD file is saved to disk at the specified location.

## Usage
1. Connect up to 10 image outputs to the layer inputs (layer1, layer2, etc.)
2. Optionally connect corresponding mask outputs to the mask inputs (mask1, mask2, etc.)
3. Set the filename prefix and output directory
4. Execute the workflow to save the PSD file

## Features
- **Fixed Input Structure**: Clearly defined inputs for up to 10 layers and masks
- **Optimized Performance**: Streamlined code for faster processing
- **Mask Integration**: Properly applies transparency masks to each layer
- **Layer Ordering**: Maintains the order of layers in the PSD file (layer1 at bottom, layer10 at top)
- **Auto-Centering**: Centers smaller images within the PSD canvas
- **Automatic Filename Management**: Prevents overwriting existing files by adding incremental numbers
- **RGB Conversion**: Ensures all layers are properly converted to RGB mode

## Differences from StarPSDSaver
- Fixed number of inputs (10 layers maximum) instead of dynamic inputs
- Pre-defined input structure for better UI organization
- Optimized code for faster processing
- Default filename prefix is "multilayer2" to distinguish from the original node

## Notes
- Layers are stacked in numerical order (layer1 at the bottom, layer10 at the top)
- If a layer has no corresponding mask, it will be saved as a fully opaque layer
- The PSD dimensions are determined by the largest connected image
- Smaller images are automatically centered within the canvas
- Masks are properly applied as layer transparency
- The node creates the output directory if it doesn't exist

This node is particularly useful for workflows that generate multiple image variations or components that need to be composed in Photoshop for further editing.
