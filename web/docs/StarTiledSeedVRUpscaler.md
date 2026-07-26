# Star Tiled SeedVR Upscaler

## Description
Upscales an image with the SeedVR2 diffusion model by processing it in overlapping tiles. This keeps VRAM usage low even for very large output resolutions. The whole SeedVR2 pipeline (preprocessing, VAE encoding, conditioning, sampling, decoding, color correction) runs inside this single node - no third-party nodes required.

## Inputs

### Required
- **image**: The image to upscale (IMAGE type)
- **model_name**: SeedVR2 diffusion model, selected from the standard `models/diffusion_models` folder (e.g. `seedvr2_7b-int8_convrot.safetensors`)
- **vae_name**: SeedVR2 VAE, selected from the standard `models/vae` folder (e.g. `ema_vae_fp16.safetensors`)
- **scale**: Upscale factor (default: 2.0, range 1.0-8.0)
- **rows**: Number of tile rows (default: 3, range 1-16). More rows = smaller tiles = less VRAM
- **cols**: Number of tile columns (default: 3, range 1-16). More columns = smaller tiles = less VRAM
- **tile_overlap**: Overlap ratio between tiles (default: 0.25, range 0.05-0.5). Higher values reduce seam artifacts but increase VRAM/time
- **color_luminance_weight**: Color transfer luminance blend weight (default: 0.8, range 0.0-1.0). Lower = more reference color matching, higher = preserve original brightness

## Outputs
- **IMAGE**: The upscaled image

## How It Works
1. The input image is upscaled by **scale** with lanczos filtering to the target resolution
2. The upscaled image is split into **rows** x **cols** tiles with **tile_overlap** overlap
3. Each tile runs through the SeedVR2 pipeline:
   - Padding to SeedVR2 requirements
   - Tiled VAE encoding
   - SeedVR2 conditioning built from the tile latent
   - Single-step Euler sampling (cfg 1.0, denoise 1.0)
   - Tiled VAE decoding
   - LAB color correction against the source tile (controlled by **color_luminance_weight**)
4. The processed tiles are blended back together with linear feathering over the overlap regions

## Tips
- More rows/columns = smaller tiles = less VRAM per step (but more steps)
- For panoramas or very wide images, increase **cols** more than **rows**
- Increase **tile_overlap** (e.g. 0.3-0.4) if you see seams between tiles; decrease it (e.g. 0.1) to save VRAM/time
- Lower **color_luminance_weight** (e.g. 0.5) for stronger color matching to the source; keep it at 0.8 for a balance
- The 7B SeedVR2 model gives the best quality; the 3B model is faster
- Batch inputs are supported; each image is processed separately

## Requirements
- A SeedVR2 diffusion model in `models/diffusion_models`
- The SeedVR2 EMA VAE in `models/vae`
- ComfyUI with native SeedVR2 support (v0.27.0 or newer)
