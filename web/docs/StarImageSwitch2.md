# Star Image Input 2 (Optimized)

## Description
An optimized version of the Star Image Input node that automatically passes the first provided input image to the output. This node serves as an automatic switch for image inputs with improved performance.

## Inputs
- **Image 1** (optional): First priority image input
- **Image 2** (optional): Second priority image input
- **Image 3** (optional): Third priority image input
- **Image 4** (optional): Fourth priority image input
- **Image 5** (optional): Fifth priority image input

## Outputs
- **img_out**: The first connected image from the inputs, or a default gray pattern if no images are connected

## Usage
Connect up to five different image sources to this node. The node will automatically select the first available image (checking inputs 1 through 5 in order) and pass it to the output. If no images are connected, it will generate a default gray patterned image.

This optimized version maintains the same functionality as the original Star Image Input node but with improved performance for complex workflows.
