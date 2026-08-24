# ⭐ Star Minimax Latent Upscaler Option — Help

Second-pass latent upscale options for the **⭐ Star Minimax All In One**
node. Instead of running anything itself, it outputs an `upscale_settings`
bundle (`UPSCALE_SETTINGS`) — connect it to the **`options` input** of the
Minimax All In One node and the whole upscale-refine group runs internally:

```
[Star Minimax Latent Upscaler Option] ──upscale_settings──> [Star Minimax All In One]
   model (optional) ────────────^                              pass 1: normal render
                                                               pass 2: upscale + refine
```

What happens inside the Minimax node when connected:

1. The pass-1 **video latent** is upscaled with the selected MiniMax H3 3D
   latent-upscaler model to the megapixel target (aspect ratio kept,
   32 px aligned, VAE-grid snapped).
2. The **audio latent** from pass 1 is carried over unchanged and the
   combined A/V latent is re-noised at the schedule's first sigma
   (≈0.90) — a light remix, not a fresh render.
3. A short **refine pass** runs with the **same conditioning and the same
   seed as pass 1** — reference image/video latents are resolution-matched
   to the upscaled canvas automatically, so `<Picture i>` / `<Video k>`
   refs keep working.
4. **IMAGE** / **LATENT** come from the refine pass; **AUDIO** comes from
   the pass selected by the audio toggle.

Everything is self-contained in the StarNodes pack — the original
`Comfyui_Minimax_h3_latent_Upscaler` / `ComfyUI-MiniMaxH3_LatentUpscaler`
packs are **not** required.

There is also a standalone twin, **⭐ Star Minimax Latent Upscaler**, which
does the same upscale + refine + decode for any workflow without the All In
One node.

## Required Model

```
models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors
```

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `model` | MODEL | optional — diffusion model for the refine pass, e.g. with a turbo LoRA (`minimax_h3_fl2v_lightx2v_turbo_4step`) and/or a sage-attention patch applied. Not connected → the pass-1 model is reused. |
| **upscale_settings** out | UPSCALE_SETTINGS | connect to the `options` input of ⭐ Star Minimax All In One |

## Widgets

- **upscale_model** — dropdown of `models/latent_upscale_models`
  (default `minimax_h3_latent_upscaler_3d_fp16`).
- **megapixels** — target size of the upscaled video (default `1.0`; the
  pass-1 canvas keeps its aspect ratio). Typical: pass 1 at 0.5 MP →
  upscale to 1.0 MP.
- **sigmas_preset** — baked refine schedules:
  - `3 steps` (default): `0.9035, 0.6316, 0.3158, 0.0`
  - `4 steps`: `0.9035, 0.8000, 0.6316, 0.3158, 0.0`
  - `5 steps`: `0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0`
- **sampler_name** — `euler` (default) matches the reference workflow and
  pairs well with a turbo LoRA on the refine model.
- **upscale_pass_audio** — `Use 1st pass audio` (default) decodes the
  soundtrack from pass 1 (untouched); `Upscale Pass Audio` decodes it after
  the refine pass, where it is lightly re-noised and rewritten.
- **align** (advanced) — pixel alignment of the upscaled size; keep `32` to
  avoid light banding.
- **enable_chunking** (advanced) — temporal chunking keeps VRAM in check on
  long videos; disable for short clips for full-context inference.
- **device** / **precision** (advanced) — `cuda`/`cpu`, `fp16` (default) /
  `fp32` / `bf16`. The upscaler model is parked back on CPU after inference
  so the refine pass gets the VRAM back.

## Notes

- The upscale pass is skipped entirely when the Minimax node's `megapixels`
  is set to `audio only`.
- The refine pass reuses the pass-1 seed — runs stay deterministic.
- The upscaler only upscales: pick a megapixel target **larger** than the
  pass-1 size (0.5 MP → 1.0 MP, not the other way around).
- Also works in `image` mode: the 9-frame still latent is upscaled and
  refined the same way before frame 8 is decoded.
