# Star Model Latent Upscaler

## Description
The Star Model Latent Upscaler is a versatile node that combines upscaling models with VAE encoding/decoding in a single workflow step. It can process images through an upscaling model, resize them to a target size, and then encode them back to latent space using a specified VAE. This node is particularly useful for high-resolution workflows and for maintaining consistency in latent space after upscaling.

## Inputs

### Required
- **VAE_OUT**: VAE model to use for encoding the upscaled image (Default or custom, including TAESD variants)
- **VAE_Device**: Device to use for VAE processing (CPU or CUDA)
- **UPSCALE_MODEL**: Upscaling model to use (Default or any installed upscale model)
- **UPSCALE_MODE**: How to determine the final resolution. Options:
  - **By Longest Side**: Match a target longest side in pixels (original behavior)
  - **By MP**: Match a target megapixel count while preserving aspect ratio
- **OUTPUT_LONGEST_SIDE**: Target size for the longest side of the output image in pixels (used when `UPSCALE_MODE` is **By Longest Side**)
- **MEGAPIXELS**: Target megapixels for the final image (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 50, 100). Used when `UPSCALE_MODE` is **By MP**
- **INTERPOLATION_MODE**: Method used for image resizing (NEAREST, BILINEAR, BICUBIC, etc.)

### Optional
- **VAE_INPUT**: Input VAE model (optional)
- **LATENT_INPUT**: Input latent tensor (optional)
- **IMAGE**: Input image tensor (optional)

## Outputs
- **vae**: The VAE model used for encoding
- **image**: The upscaled and resized image
- **latent**: The latent representation of the upscaled image
- **width**: Final image width as a string (useful for string-based nodes)
- **height**: Final image height as a string

## Usage
This node can be used in three different ways:

1. **Image Upscaling**: Connect an image to the IMAGE input to upscale and resize it
2. **Latent Decoding + Upscaling**: Connect both a VAE and latent to VAE_INPUT and LATENT_INPUT to decode, upscale, and re-encode
3. **Standalone VAE Provider**: Use without inputs to get a configured VAE for use elsewhere in your workflow

## Features
- **Integrated Workflow**: Combines upscaling, resizing, and VAE encoding in a single node
- **Multiple Input Paths**: Flexibly handles different input types (image or latent)
- **TAESD Support**: Built-in support for TAESD, TAESDXL, TAESD3, and TAEF1 approximate VAEs
- **Device Control**: Explicit control over which device (CPU/GPU) the VAE uses
- **Smart Resizing**: Maintains aspect ratio while targeting a specific longest side dimension or target megapixels
- **Dimension Correction**: Ensures output dimensions are divisible by 16 for compatibility with diffusion models

## Technical Details
- The node automatically detects which inputs are connected and chooses the appropriate processing path
- When upscaling, the image is first processed through the selected upscale model (if not "Default")
- The image is then resized to match either the specified longest side or the target megapixel count while maintaining aspect ratio
- Both dimensions are adjusted to be divisible by 16 for proper latent space compatibility
- Finally, the image is encoded to latent space using the specified VAE

## Notes
- For best results with specific model architectures, use the matching VAE (e.g., TAESDXL for SDXL models)
- When using **By Longest Side**, the `OUTPUT_LONGEST_SIDE` parameter will be automatically adjusted to be divisible by 16
- When using **By MP**, the node computes a scale factor from the requested megapixels (1 MP = 1,000,000 pixels), then rounds the final width and height to the nearest multiple of 16
- If no inputs are connected, the node will still provide the configured VAE and a default black image
- This node is particularly useful for workflows that require high-resolution processing while maintaining latent space consistency
