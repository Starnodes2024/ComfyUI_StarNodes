# Starnodes Aspect Ratio Advanced

## Description
An enhanced version of the Starnodes Aspect Ratio node that not only calculates dimensions based on aspect ratios but also generates empty latent tensors compatible with different model architectures (SDXL/Flux and SD3.5). This node helps streamline workflows by providing both dimensions and properly sized latent tensors in a single node.

## Inputs
- **aspect_ratio**: Select from predefined aspect ratios (e.g., "16:9", "4:3", "1:1", etc.)
- **megapixel**: Target image size in megapixels (options: 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
- **use_nearest_from_image**: When enabled, the node will analyze the input image and select the closest matching aspect ratio
- **batch_size**: Number of empty latent tensors to generate (default: 1)
- **image** (optional): Input image for aspect ratio detection when `use_nearest_from_image` is enabled

## Outputs
- **width**: The calculated width in pixels
- **height**: The calculated height in pixels
- **Resolution**: A formatted string showing the resolution (e.g., "512 x 768")
- **SDXL / FLUX**: Empty latent tensor with shape [batch_size, 4, height/8, width/8] for SDXL and Flux models
- **SD3.5**: Empty latent tensor with shape [batch_size, 4, height/16, width/16] for SD3.5 models

## Usage
1. Select a desired aspect ratio from the dropdown menu
2. Choose the target megapixel size for your output image
3. Set the batch size for the number of latent tensors to generate
4. Optionally, connect an image and enable `use_nearest_from_image` to automatically match the closest standard aspect ratio
5. Use the output width and height values to configure image generation nodes
6. Connect the appropriate latent output to compatible model nodes (SDXL/Flux or SD3.5)

This node is particularly useful for:
- Creating workflows that require both dimension values and empty latent tensors
- Ensuring proper latent tensor sizes for different model architectures
- Generating batches of empty latents with the correct dimensions
- Streamlining workflows by combining dimension calculation and latent tensor creation

The node ensures that the generated latent tensors have dimensions properly adjusted for the downsampling factors of different model architectures (8x for SDXL/Flux and 16x for SD3.5).
