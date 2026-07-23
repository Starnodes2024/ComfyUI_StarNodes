# Star Tiled PiD Upscaler

## Description
Upscales an image with a PiD / PixelDiT model by processing it in overlapping tiles to keep VRAM usage low. Each tile is re-rendered in pixel space (4 LCM steps) and blended back seamlessly, with per-tile color matching.

PiD is a latent-conditioned *pixel-space* diffusion upscaler: the low-res image is VAE-encoded and injected into the diffusion model ("lq_latent"), which then re-renders the image directly in pixel space at the requested output size in ~4 distilled steps (no VAE decode needed — a virtual "pixel_space" VAE is used).

This node replicates the reference ComfyUI workflow (UNETLoader + CLIPLoader(pixeldit) + VAEEncode + PiDConditioning + ManualSigmas + LCM SamplerCustom + pixel_space VAEDecode + ColorTransfer) but processes the image in overlapping tiles so VRAM usage stays low and much bigger upscales are possible.

## Inputs

### Required
- **image**: The image to upscale (IMAGE type)
- **model_name**: The PiD / PixelDiT diffusion model, selected from `models/diffusion_models` (e.g. `pid_qwenimage_1024_to_4096_4step_bf16.safetensors`)
- **clip_name**: Gemma 2 2B text encoder for PixelDiT (`gemma_2_2b_it_elm`). PiD uses an empty prompt, but the model still needs the text encoder loaded
- **vae_name**: VAE matching the PiD backbone, used to encode the input image (e.g. `qwen_image_vae` for qwenimage, `ae.safetensors` for flux)
- **latent_format**: Latent format of the PiD backbone. Options: `qwenimage`, `flux`, `sd3`, `sdxl`. Flux1 (16-ch) and Flux2 (128-ch) are auto-detected under 'flux'
- **scale**: Upscale factor for the output image (default: 4.0, range 1.0-8.0). PiD models are trained for 4x
- **rows**: Number of tile rows (default: 2, range 1-16). More rows = smaller tiles = less VRAM
- **cols**: Number of tile columns (default: 2, range 1-16). More columns = smaller tiles = less VRAM

## Outputs
- **IMAGE**: The upscaled image

## How It Works
1. The input image is divided into **rows** x **cols** tiles with 10% overlap
2. Each tile runs through the PiD pipeline:
   - VAE encoding with the backbone VAE (qwen_image_vae, ae.safetensors, etc.)
   - PiD conditioning built from the tile latent (lq_latent + degrade_sigma)
   - 4-step LCM sampling in pixel space with manual sigmas
   - Pixel-space VAE decoding (virtual "pixel_space" VAE)
   - LAB color transfer to match the source tile colors
3. The processed tiles are blended back together with feathered masks over the overlap regions
4. For Flux2 PiD models (128-channel latents), an additional calibrated color/brightness bias correction is applied

## Tips
- More rows/columns = smaller tiles = less VRAM per step (but more steps)
- For panoramas or very wide images, increase **cols** more than **rows**
- The node automatically handles different PiD backbones (qwenimage, flux, sd3, sdxl) based on the **latent_format** setting
- Batch inputs are supported; each image is processed separately
- Tiles are color-matched to the source image using wavelet + LAB histogram transfer, which replaces both the PiD color bias patch and the final ColorTransfer node of the manual workflow

## Requirements
- A PiD / PixelDiT diffusion model in `models/diffusion_models`
- The Gemma 2 2B text encoder (`gemma_2_2b_it_elm`) in `models/text_encoders`
- The appropriate VAE for your PiD backbone in `models/vae`
- ComfyUI with native PixelDiT/PiD support (PiDConditioning core node / comfy.ldm.pixeldit)

## Example Models
- **QwenImage PiD**: `pid_qwenimage_1024_to_4096_4step_bf16.safetensors` + `qwen_image_vae` + latent_format: `qwenimage`
- **Flux PiD**: `pid_flux_*` + `ae.safetensors` + latent_format: `flux`
- **SD3 PiD**: `pid_sd3_*` + SD3 VAE + latent_format: `sd3`
- **SDXL PiD**: `pid_sdxl_*` + SDXL VAE + latent_format: `sdxl`
