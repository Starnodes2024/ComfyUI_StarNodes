# Star Image2Latent

## Description
This node converts an input image to a latent representation using a VAE (Variational Autoencoder). It provides options to add configurable noise and supports different latent types for compatibility with various models (SD, SDXL, SD3.5, and FLUX).

## Inputs
- **image**: The input image to convert to latent space
- **vae**: The VAE model to use for encoding
- **noise_amount**: Amount of noise to add to the latent (range: 0.0 to 1.0, default: 0.0)
- **latent_type**: The model type to target, which determines expected channel dimensions:
  - `SD`: Standard Stable Diffusion (4 channels)
  - `SDXL`: Stable Diffusion XL (4 channels)
  - `SD3.5`: Stable Diffusion 3.5 (4 channels)
  - `FLUX`: Flux models (8 channels)

## Outputs
- **latent**: The encoded latent representation of the input image with optional added noise

## Usage
1. Connect an image source to the input
2. Connect a VAE model compatible with your target model type
3. Adjust the noise amount if you want to add variation to the latent
4. Select the appropriate latent type based on the model you're using

This node is particularly useful for:
- Converting images to latent space for further processing
- Adding controlled noise to latents for variation
- Ensuring proper channel dimensions for different model types
- Creating custom inpainting or img2img workflows

Note: The node will display a warning if the VAE produces a latent with a different number of channels than expected for the selected model type.
