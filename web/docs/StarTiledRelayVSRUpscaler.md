# Star Tiled RelayVSR Upscaler

## Description
Upscales an image with the RelayVSR diffusion model by processing it in overlapping tiles. This keeps VRAM usage low even for very large output resolutions. The whole RelayVSR pipeline (conditioning, one-step sampling, VSR flash decoding) runs inside this single node - all code is inlined, so the node does not load the native RelayVSR core nodes and keeps working even after core changes.

## Inputs

### Required
- **image**: The image to upscale (IMAGE type). It goes in at its native size - the RelayVSR checkpoint decides the upscale factor (e.g. x4)
- **model_name**: RelayVSR diffusion model, selected from the standard `models/diffusion_models` folder (e.g. `relayvsr_x4_int8_convrot.safetensors`)
- **vae_name**: RelayVSR Flash VAE, selected from the standard `models/vae` folder (e.g. `relayvsr_flash_vae_x4_fp32.safetensors`)
- **seed**: Random seed for the sampling pass (default: 0, control set to `fixed` so re-queued runs stay reproducible)
- **tiled**: Toggle (default: on). On: the image is split into overlapping tiles to keep VRAM low - recommended for large images. Off: the whole image is processed in a single pass - faster and seam-free, but needs a lot of VRAM on large images
- **max_tile_size**: Maximum tile width/height in pixels of the input image (default: 1024, range 64-16384). The image is split into as many tiles as needed to stay under this size - smaller tiles = less VRAM, larger tiles = fewer seams. Ignored while **tiled** is off
- **tile_overlap**: Overlap ratio between tiles (default: 0.25, range 0.05-0.5). Higher values reduce seam artifacts but increase VRAM/time. Ignored while **tiled** is off
- **detail**: Slider. Scales the high-frequency residual the model adds on top of the plain bicubic upscale of each frame (default: 0.6, range 0.0-1.0). 0.0 = pure bicubic upscale (no model detail), 1.0 = full model output
- **batch_size**: Frames processed together per tile (default: 0 = auto). Auto runs the whole input as ONE video pass - the fastest setting, since the model only samples every 15th frame anyway and the decoder restores all frames. 1 = each image separately (use this for unrelated images). >1 = fixed frames per pass (lower VRAM than auto on very long videos)

### Optional
- **model_override**: Optional MODEL input. When connected, this model will be used instead of the one selected in the **model_name** dropdown. This is useful for:
  - Using patched models from other nodes
  - Using models loaded with custom loaders
  - Applying model patches or modifications before upscaling

## Outputs
- **IMAGE**: The upscaled image (native size x checkpoint factor, e.g. x4)

## How It Works
1. With **tiled** on, the input image is split into overlapping tiles so that no tile exceeds **max_tile_size** pixels per side (with **tile_overlap** overlap), all at native resolution. With **tiled** off, the whole image is processed as a single tile
2. Each tile runs through the RelayVSR pipeline:
   - RelayVSR conditioning built from the tile (the fixed prompt lives in the checkpoint)
   - Single-pass sampling (1 step, cfg 1.0, Euler / simple, denoise 1.0)
   - VSR flash decoding with the low-quality tile as reference and the **detail** blend
   - The decoded tile is aligned onto the factor lattice for a seamless merge
3. The processed tiles are blended back together with linear feathering over the overlap regions (scaled by the upscale factor)

## Tips
- Lower **max_tile_size** when you run out of VRAM (e.g. 512); raise it (e.g. 1536-2048) for fewer tiles and fewer seams. Works with very large images - the tile count grows automatically
- Increase **tile_overlap** (e.g. 0.3-0.4) if you see seams between tiles; decrease it (e.g. 0.1) to save VRAM/time
- Lower **detail** (e.g. 0.6-0.8) if the model adds too much texture or noise for your source; 0.0 gives you the plain bicubic upscale; 1.0 is the full model output
- RelayVSR restores detail from the low-quality source, so colors stay close to the original - no color-fix pass is needed
- For videos leave **batch_size** on auto (0): a 400-frame batch becomes a single pass where only ~27 keyframes go through the model. If you run out of VRAM on very long videos, set a fixed batch_size (e.g. 64-128) to chunk it
- For unrelated images in one batch (not a video), set **batch_size** to 1 so each image is processed as a still image instead of being blended like video frames
- Higher **batch_size** uses more VRAM; combine it with **tiled** off for the best quality when memory allows
- Use **model_override** to connect patched models from LoRA loaders, model patchers, or other custom model processing nodes. When **model_override** is connected, the **model_name** dropdown is ignored

## Requirements
- A RelayVSR diffusion model in `models/diffusion_models` (e.g. `relayvsr_x4_int8_convrot.safetensors`)
- The RelayVSR Flash VAE in `models/vae` (e.g. `relayvsr_flash_vae_x4_fp32.safetensors`)
- ComfyUI with native RelayVSR model support - the node itself is self-contained and does not use the native RelayVSR nodes
