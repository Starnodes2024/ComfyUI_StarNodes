# ⭐ Star SD Upscale Refiner Advanced

All-in-one upscale + refine node taking direct `MODEL` / `CLIP` / `VAE` inputs, so it
works with any checkpoint loader output — SD1.5, SDXL, Flux, Fibo2, Qwen and other
flow models.

Category: `⭐StarNodes/Upscale`

## Inputs

| Input | Description |
|---|---|
| `model` / `clip` / `vae` | Direct connectors (e.g. from Load Checkpoint) |
| `lora_1..3_name` + strengths | Optional LoRAs baked onto the refine pass |
| `positive_prompt` / `negative_prompt` / `use_negative` | Internal encoding; ignored while `pos_condition` / `neg_condition` are connected |
| `seed`, `refine_steps`, `refine_denoise` | Sampling controls. `refine_denoise` ~0.2 is typical |
| `cfg` | 1.0 for flow models (Flux, Fibo2, ...). SD/SDXL can go higher, AutoCFG normalizes |
| `tile_size` / `refine_tiled` | Tiled refine (low VRAM) vs whole-latent refine (auto-falls back to tiled above 2 MP) |
| `free_vram` / `cuda_sync_guard` | Unload models before run / per-tile sync debugging |
| `sampler_name` / `scheduler_name` | ComfyUI sampler + scheduler for the refine pass |
| `IMAGE`, `pos_condition`, `neg_condition`, `controlnet_name` | Optional connectors |
| `UPSCALE_IMAGE` / `UPSCALE_MODEL` / `OUTPUT_LONGEST_SIDE` | Optional model upscale + target longest side |
| `options` | ⭐ Star Split Sampler Option or ⭐ Star Custom Sigmas |

## Outputs

- `image` (IMAGE): refined result
- `latent` (LATENT): refined latent for downstream chains

## Model support notes

- **Flow models (Flux, Fibo2, Qwen, SD3, Wan image, ...)**: the sigma schedule is built from the
  model's own `model_sampling` object exactly like the native KSampler, honoring `refine_denoise`.
  Keep `cfg` at 1.0 there.
- **k-diffusion models (SD1.5, SDXL, Cascade, ...)**: the refine pass uses AlignYourSteps (SD1)
  scheduling; `cfg` > 1 is supported via AutoCFG normalization.
- **Tiled refine** reads the real VAE downscale ratio (`vae.downscale_ratio`), so 16× VAEs
  (Fibo2/Flux2, 128-channel latents) get correct tile footprints.
- For older-style behavior notes see `StarSDUpscaleRefiner.md` (SD1.5 checkpoint variant).
