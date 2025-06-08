# StarSampler SD/SDXL/SD3.5

## Description
The StarSampler SD/SDXL/SD3.5 is a versatile sampler node designed to work with all major Stable Diffusion model versions (SD 1.5, SDXL, SD 3.5). It features an integrated detail enhancement system based on Detail Daemon technology, which can improve fine details in generated images by dynamically adjusting the noise schedule during the sampling process.

## Inputs

### Required
- **model**: The diffusion model to use (SD 1.5, SDXL, or SD 3.5)
- **positive**: Positive conditioning (from CLIP text encoder)
- **negative**: Negative conditioning (from CLIP text encoder)
- **latent**: Latent noise input
- **seed**: Random seed for reproducible results
- **steps**: Number of sampling steps
- **cfg**: Classifier-free guidance scale (how closely to follow the prompt)
- **sampler_name**: Sampling algorithm (euler, euler_ancestral, heun, dpm_2, etc.)
- **scheduler**: Noise schedule (normal, karras, exponential, etc.)
- **denoise**: Denoising strength (1.0 for full sampling, lower for img2img)
- **vae**: VAE model for decoding latents to images
- **decode_image**: Whether to decode the latent to an image (boolean)

### Optional
- **detail_schedule**: Detail enhancement schedule from a detail scheduler node
- **settings_input**: Previous settings from another StarSampler node

## Outputs
- **model**: The model after sampling (unchanged)
- **positive**: The positive conditioning (unchanged)
- **negative**: The negative conditioning (unchanged)
- **latent**: The sampled latent
- **detail_schedule**: The detail schedule used (for chaining)
- **image**: The decoded image (if decode_image is true)
- **vae**: The VAE model (unchanged)
- **settings_output**: The settings used for sampling
- **seed**: The seed used for sampling

## Usage
1. Connect a model (SD 1.5, SDXL, or SD 3.5) to the model input
2. Connect positive and negative conditioning from CLIP Text Encode nodes
3. Connect a latent source (empty or noisy latent)
4. Set sampling parameters (steps, cfg, sampler, scheduler, denoise)
5. Optionally connect a detail schedule for enhanced details
6. Connect the outputs to further processing nodes or image savers

## Features

### Detail Enhancement
When connected to a detail schedule, the sampler can enhance specific frequency details:
- Dynamically adjusts the noise schedule during sampling
- Preserves and enhances fine details that might otherwise be lost
- Particularly effective for faces, textures, and intricate patterns
- Works with all supported model architectures (SD 1.5, SDXL, SD 3.5)

### Universal Compatibility
Unlike some specialized samplers, the StarSampler works with all major Stable Diffusion model versions:
- Standard SD 1.5 models
- SDXL base and refiner models
- SD 3.5 models
- Compatible with various VAE types

### Efficient Processing
- Uses ComfyUI's common_ksampler for optimal performance
- Temporarily modifies the model's forward pass only when detail enhancement is active
- Restores original model behavior after sampling

## Technical Details
- Detail enhancement works by dynamically adjusting sigma values during the diffusion process
- The adjustment follows a cosine-based schedule with configurable start, mid, and end points
- Detail strength is modulated by the CFG value, creating stronger effects at higher guidance scales
- The implementation carefully handles edge cases to ensure stability across different model types

## Notes
- For best results with detail enhancement, use 20+ sampling steps
- Detail enhancement works best with the "euler" sampler and "normal" or "karras" scheduler
- The detail schedule can be created using a DetailStarDaemon node
- When using with SD 3.5 models, slightly lower CFG values (5-7) often produce better results
- The settings_output can be connected to another StarSampler node to maintain consistent settings
