# Star SD Upscale Refiner

## Description
The Star SD Upscale Refiner is a comprehensive all-in-one node that combines the entire SD1.5 upscaling and refinement workflow into a single, easy-to-use node. It integrates checkpoint loading, multiple LoRAs, upscale models, tiled processing, advanced sampling techniques, and ControlNet tile processing - eliminating the need for complex node chains.

This node is specifically designed for high-quality image upscaling and refinement, incorporating advanced techniques like FreeU, Perturbed Attention Guidance (PAG), Automatic CFG, and Tiled Diffusion for optimal results.

## Key Features
- **All-in-One Solution**: Complete upscaling workflow in a single node
- **No External Dependencies**: All functionality is built-in
- **Multiple LoRA Support**: Load up to 3 LoRAs with optional "None" setting
- **Advanced Upscaling**: Combines upscale models with intelligent resizing
- **Tiled Processing**: Memory-efficient VAE encoding/decoding and diffusion
- **Built-in Optimizations**: FreeU, PAG, and Automatic CFG automatically applied
- **ControlNet Integration**: Tile-based ControlNet for coherent upscaling
- **Flexible Output Sizing**: Specify target size by longest side

## Inputs

### Image Input
- **image**: Input image to upscale and refine

### Model Settings
- **checkpoint_name**: SD1.5 checkpoint model to use for refinement
  - Default: `SD1.5\juggernaut_reborn.safetensors`
  - Choose your preferred SD1.5 model

### LoRA Settings (Optional)
All three LoRAs default to "None" and are only applied if you select a model:

- **lora_1_name**: First LoRA model (default: "None")
- **lora_1_strength**: Strength for first LoRA (default: 0.25)
  - Applied to both model and CLIP
  - Range: -10.0 to 10.0

- **lora_2_name**: Second LoRA model (default: "None")
- **lora_2_strength**: Strength for second LoRA (default: 0.1)

- **lora_3_name**: Third LoRA model (default: "None")
- **lora_3_strength**: Strength for third LoRA (default: 0.1)

### Upscaling Settings
- **upscale_model_name**: Upscale model to use
  - Example: `4xNomosUniDAT_otf.pth`
  - Applies AI-based upscaling before refinement

- **output_longest_side**: Target size for the longest side in pixels
  - Default: 2048
  - Range: 512 to 16384 (step: 64)
  - The image is scaled so its longest side matches this value
  - Aspect ratio is preserved
  - Uses high-quality bicubic interpolation (hardcoded for best results)

### Prompt Settings
- **positive_prompt**: Positive prompt for refinement
  - Default: "masterpiece, best quality, highres"
  - Multiline text input
  - Describes desired qualities to enhance

- **negative_prompt**: Negative prompt for refinement
  - Default: "(worst quality, low quality, normal quality:1.5)"
  - Multiline text input
  - Describes qualities to avoid

### ControlNet Settings
- **controlnet_name**: ControlNet model for tile control
  - Default: `control_v11f1e_sd15_tile.pth`
  - Ensures coherent upscaling across tiles

- **controlnet_strength**: ControlNet conditioning strength
  - Default: 0.5
  - Range: 0.0 to 1.0
  - Higher values = stronger tile guidance

### Tiled Diffusion Settings
These settings control how the image is processed in tiles for memory efficiency:

- **tile_width**: Width of each tile for tiled diffusion
  - Default: 1024
  - Range: 256 to 4096 (step: 64)

- **tile_height**: Height of each tile for tiled diffusion
  - Default: 1024
  - Range: 256 to 4096 (step: 64)

- **tile_overlap**: Overlap between tiles in pixels
  - Default: 128
  - Range: 0 to 512 (step: 8)
  - Higher overlap = better blending but slower

- **tile_batch_size**: Number of tiles to process in parallel
  - Default: 4
  - Range: 1 to 16
  - Higher = faster but more VRAM usage

### Sampling Settings
- **sampler_name**: Sampler algorithm
  - Default: dpmpp_3m_sde_gpu
  - All ComfyUI samplers available

- **scheduler**: Scheduler type
  - Default: SD1 (AlignYourSteps)
  - Optimized for SD1.5 models

- **steps**: Number of sampling steps
  - Default: 18
  - Range: 1 to 150
  - More steps = higher quality but slower

- **denoise**: Denoising strength
  - Default: 0.5
  - Range: 0.0 to 1.0
  - 0.5 recommended for refinement (preserves original image)
  - Higher values = more changes to the image

- **cfg**: Classifier-free guidance scale
  - Default: 8.0
  - Range: 0.0 to 30.0
  - Higher = stronger prompt adherence

- **seed**: Random seed for sampling
  - Default: 0
  - Range: 0 to max int
  - Use same seed for reproducible results

- **seed_mode**: Seed generation mode
  - Options: fixed, random, increment
  - fixed: Use the specified seed
  - random: Generate new random seed each time
  - increment: Increase seed by 1 each time

### VAE Settings
- **vae_tile_size**: Tile size for VAE encoding/decoding
  - Default: 1024
  - Range: 512 to 4096 (step: 64)
  - Larger = faster but more VRAM usage

## Outputs
- **refined_image**: The upscaled and refined output image

## Built-in Optimizations

### FreeU V2 (Automatic)
Applied automatically with fixed parameters:
- Backbone factor 1: 1.05
- Backbone factor 2: 1.08
- Skip factor 1: 0.95
- Skip factor 2: 0.8

Improves image quality and reduces artifacts.

### Perturbed Attention Guidance (Automatic)
Applied automatically with scale: 1.0

Enhances detail and coherence in generated images.

### Automatic CFG (Automatic)
Applied automatically with:
- Subtract mean: True
- Fake uncond: True

Optimizes classifier-free guidance for better results.

## Workflow Process
The node executes the following steps internally:

1. **Load checkpoint** and VAE
2. **Apply LoRAs** (only if selected, not "None")
3. **Upscale image** with selected upscale model
4. **Scale to target size** based on longest side (bicubic interpolation)
5. **Encode to latent** using tiled VAE encoding
6. **Apply FreeU** optimization to model
7. **Encode prompts** with CLIP
8. **Apply ControlNet** for tile coherence
9. **Apply PAG** optimization to model
10. **Apply Automatic CFG** optimization to model
11. **Apply Tiled Diffusion** for memory-efficient processing
12. **Sample** with specified settings
13. **Decode to image** using tiled VAE decoding

## Usage Tips

### For Best Quality
- Use a high-quality upscale model like 4xNomosUniDAT
- Set denoise to 0.4-0.6 for refinement (preserves original)
- Use 18-25 steps for good quality/speed balance
- Enable ControlNet tile for coherent upscaling

### For Memory Efficiency
- Reduce tile_width and tile_height (e.g., 768x768)
- Increase tile_overlap for better blending
- Reduce tile_batch_size if running out of VRAM
- Reduce vae_tile_size if VAE encoding/decoding fails

### For Speed
- Use fewer steps (12-18)
- Increase tile_batch_size (if VRAM allows)
- Use faster samplers like euler or dpmpp_2m

### LoRA Usage
- All LoRAs default to "None" - only applied if you select one
- Use detail LoRAs (0.2-0.4 strength) for enhanced details
- Use style LoRAs (0.1-0.3 strength) for artistic refinement
- Stack multiple LoRAs for combined effects

## Example Settings

### High-Quality 2K Upscale
- checkpoint: SD1.5\juggernaut_reborn.safetensors
- output_longest_side: 2048
- upscale_model: 4xNomosUniDAT_otf.pth
- steps: 20
- denoise: 0.5
- cfg: 8.0

### Fast 1080p Upscale
- checkpoint: SD1.5\juggernaut_reborn.safetensors
- output_longest_side: 1920
- upscale_model: 4xNomosUniDAT_otf.pth
- steps: 12
- denoise: 0.4
- cfg: 7.0

### Maximum Quality 4K Upscale
- checkpoint: SD1.5\juggernaut_reborn.safetensors
- output_longest_side: 3840
- upscale_model: 4xNomosUniDAT_otf.pth
- steps: 25
- denoise: 0.6
- cfg: 8.5
- tile_overlap: 192

## Technical Notes
- All dimensions are automatically adjusted to be divisible by 64
- Tiled processing prevents VRAM overflow on large images
- FreeU, PAG, and Automatic CFG are applied with optimized fixed parameters
- ControlNet tile ensures coherent upscaling without seams
- AlignYourSteps scheduler provides optimal noise scheduling for SD1.5

## Troubleshooting

### Out of Memory Errors
- Reduce tile_width and tile_height
- Reduce tile_batch_size
- Reduce vae_tile_size
- Reduce output_longest_side

### Tile Seams Visible
- Increase tile_overlap (try 192 or 256)
- Increase controlnet_strength
- Ensure tile dimensions are appropriate for image size

### Image Too Different from Original
- Reduce denoise (try 0.3-0.4)
- Reduce cfg (try 6.0-7.0)
- Adjust positive_prompt to be less specific

### Processing Too Slow
- Reduce steps
- Reduce tile_overlap
- Increase tile_batch_size (if VRAM allows)
- Use faster sampler

## Category
⭐StarNodes/Upscale

## Version
1.0.0
