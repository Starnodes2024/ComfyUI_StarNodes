# Star Image Input

## Description
This node acts as an automatic switch for image inputs. It passes the first connected image to the output, allowing for flexible workflow design where multiple potential image sources can be connected without manual switching.

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

This node is useful for:
- Creating workflows with alternative image sources
- Building conditional image processing pipelines
- Simplifying complex workflows by reducing the need for manual switching between image sources
