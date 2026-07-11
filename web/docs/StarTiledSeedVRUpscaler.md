# Star Tiled SeedVR Upscaler

## Description
Upscales an image with the SeedVR2 diffusion model by processing it in overlapping tiles. This keeps VRAM usage low even for very large output resolutions. The whole SeedVR2 pipeline (preprocessing, VAE encoding, conditioning, sampling, decoding, color correction) runs inside this single node - no third-party nodes required.

## Inputs

### Required
- **image**: The image to upscale (IMAGE type)
- **model_name**: SeedVR2 diffusion model, selected from the standard `models/diffusion_models` folder (e.g. `seedvr2_7b-int8_convrot.safetensors`)
- **vae_name**: SeedVR2 VAE, selected from the standard `models/vae` folder (e.g. `ema_vae_fp16.safetensors`)
- **scale**: Upscale factor (default: 2.0, range 1.0-8.0)
- **rows**: Number of tile rows (default: 3)
- **cols**: Number of tile columns (default: 3)

## Outputs
- **IMAGE**: The upscaled image

## How It Works
1. The input image is upscaled by **scale** with lanczos filtering to the target resolution
2. The upscaled image is split into **rows** x **cols** tiles with 10% overlap
3. Each tile runs through the SeedVR2 pipeline:
   - Padding to SeedVR2 requirements
   - Tiled VAE encoding
   - SeedVR2 conditioning built from the tile latent
   - Single-step Euler sampling (cfg 1.0, denoise 1.0)
   - Tiled VAE decoding
   - LAB color correction against the source tile
4. The processed tiles are blended back together with linear feathering over the overlap regions

## Tips
- More rows/columns = smaller tiles = less VRAM per step (but more steps)
- For panoramas or very wide images, increase **cols** more than **rows**
- The 7B SeedVR2 model gives the best quality; the 3B model is faster
- Batch inputs are supported; each image is processed separately

## Requirements
- A SeedVR2 diffusion model in `models/diffusion_models`
- The SeedVR2 EMA VAE in `models/vae`
- ComfyUI with native SeedVR2 support (v0.27.0 or newer)
