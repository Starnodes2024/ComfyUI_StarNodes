# StarSampler FLUX

## Description
The StarSampler FLUX is an advanced sampler node specifically optimized for Flux models in ComfyUI. It extends standard sampling with features like detail enhancement, multi-step scheduling, and TEA-Cache acceleration. This sampler provides fine-grained control over the sampling process, allowing for better detail preservation, faster generation, and more consistent results when working with Flux architecture models.

## Inputs

### Required
- **model**: The model to use for sampling (typically a Flux model)
- **conditioning**: Positive and negative conditioning for the generation
- **latent**: The latent input to be sampled
- **seed**: Random seed to control generation reproducibility
- **sampler**: Sampling algorithm to use (euler, euler_ancestral, heun, dpm_2, dpm_2_ancestral, etc.)
- **scheduler**: Scheduler to use (normal, karras, exponential, sgm_uniform, etc.)
- **steps**: Number of sampling steps (supports comma-separated lists for multi-step sampling)
- **guidance**: CFG scale value (supports comma-separated lists for multi-step sampling)
- **max_shift**: Maximum shift value for TEA-Cache acceleration (leave empty to disable)
- **base_shift**: Base shift value for TEA-Cache acceleration (leave empty to disable)
- **denoise**: Denoising strength (supports comma-separated lists for multi-step sampling)
- **vae**: VAE model to decode latents into images
- **decode_image**: Whether to decode the latent to an image using the VAE

### Optional
- **detail_schedule**: Detail enhancement schedule from a detail scheduler node
- **settings_input**: Previous settings from another StarSampler FLUX node

## Outputs
- **model**: The model after sampling (unchanged)
- **conditioning**: The conditioning after sampling (unchanged)
- **latent**: The sampled latent
- **detail_schedule**: The detail schedule used (for chaining to other nodes)
- **image**: The decoded image (if decode_image is true)
- **vae**: The VAE model (unchanged)
- **settings_output**: The settings used for sampling (can be fed into another sampler)
- **seed**: The seed used for sampling

## Usage
1. Connect a Flux model to the model input
2. Connect conditioning from a conditioning node
3. Connect a latent source (empty or noisy latent)
4. Set sampling parameters (steps, guidance, denoise)
5. Optionally enable TEA-Cache by setting max_shift and base_shift
6. Optionally connect a detail schedule for enhanced details
7. Connect the outputs to further processing nodes or image savers

## Features

### Multi-Step Sampling
The node supports comma-separated lists for steps, guidance, and denoise parameters, allowing for multi-step sampling with different parameters at each step. For example:
- steps: "20,10" - 20 steps followed by 10 steps
- guidance: "7.5,3.0" - CFG 7.5 for first step, 3.0 for second step
- denoise: "0.6,1.0" - Partial denoise first, then full denoise

### TEA-Cache Acceleration
TEA-Cache (Temporal Attention Efficiency Cache) can significantly speed up sampling by reusing attention computations:
- **max_shift**: Maximum threshold for attention caching (recommended: 0.05-0.1)
- **base_shift**: Base threshold for attention caching (recommended: 0.01-0.03)
- Higher values = faster sampling but potentially lower quality

### Detail Enhancement
When connected to a detail schedule, the sampler can enhance specific frequency details:
- Uses a modified diffusion process to emphasize details at specific sampling steps
- Particularly effective for Flux models which can sometimes lose fine details

## Technical Details
- The sampler is based on ComfyUI's k-sampler system with additional optimizations
- Detail enhancement is adapted from the Detail Daemon project
- TEA-Cache implementation is optimized for Flux model architecture
- The node handles batched latents for generating multiple images in one pass

## Notes
- For best results with Flux models, use the "euler" sampler with "normal" or "karras" scheduler
- When using TEA-Cache, start with low values (max_shift: 0.05, base_shift: 0.01) and adjust as needed
- Multi-step sampling can be used to create a coarse-to-fine approach (high CFG first, lower CFG later)
- The settings_output can be connected to another StarSampler FLUX node to maintain consistent settings
- Detail enhancement works best when applied to the middle range of the sampling process (0.3-0.7)
