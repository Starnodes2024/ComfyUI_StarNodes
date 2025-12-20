# Starnodes Aspect Ratio Advanced

## Description
An enhanced version of the Starnodes Aspect Ratio node that not only calculates dimensions based on aspect ratios but also generates empty latent tensors compatible with different model architectures (SDXL/Flux, SD3.5, and Flux 2). This node helps streamline workflows by providing both dimensions and properly sized latent tensors in a single node.

## Inputs
- **aspect_ratio**: Select from predefined aspect ratios (e.g., "16:9", "4:3", "1:1", etc.)
- **megapixel**: Target image size in megapixels (options: 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, up to 10.0)
- **latent_channels**: Select the latent format for your model type:
  - **SDXL / FLUX (4ch)**: 4 channels, 8x downsampling - for SDXL and Flux 1.x models
  - **SD3.5 (4ch)**: 4 channels, 8x downsampling - for SD3.5 models
  - **FLUX 2 (128ch)**: 128 channels, 16x downsampling - for Flux 2 models
- **use_nearest_from_image**: When enabled, the node will analyze the input image and select the closest matching aspect ratio
- **batch_size**: Number of empty latent tensors to generate (default: 1)
- **image** (optional): Input image for aspect ratio detection when `use_nearest_from_image` is enabled

## Outputs
- **width**: The calculated width in pixels
- **height**: The calculated height in pixels
- **Resolution**: A formatted string showing the resolution (e.g., "512 x 768")
- **latent**: Empty latent tensor with shape determined by the `latent_channels` selection

## Usage
1. Select a desired aspect ratio from the dropdown menu
2. Choose the target megapixel size for your output image
3. **Select the latent_channels option** that matches your model type:
   - Choose **SDXL / FLUX (4ch)** for SDXL or Flux 1.x models
   - Choose **SD3.5 (4ch)** for SD3.5 models
   - Choose **FLUX 2 (128ch)** for Flux 2 models
4. Set the batch size for the number of latent tensors to generate
5. Optionally, connect an image and enable `use_nearest_from_image` to automatically match the closest standard aspect ratio
6. Use the output width and height values to configure image generation nodes
7. Connect the latent output to your model's latent input

This node is particularly useful for:
- Creating workflows that require both dimension values and empty latent tensors
- Ensuring proper latent tensor sizes for different model architectures
- Generating batches of empty latents with the correct dimensions
- Streamlining workflows by combining dimension calculation and latent tensor creation

The node automatically generates the correct latent tensor format based on your selection:
- **SDXL / FLUX (4ch)**: 4 channels, 8x downsampling
- **SD3.5 (4ch)**: 4 channels, 8x downsampling
- **FLUX 2 (128ch)**: 128 channels, 16x downsampling
