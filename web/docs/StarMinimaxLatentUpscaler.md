# ⭐ Star Minimax Latent Upscaler — Help

Standalone second-pass latent upscale + refine for **MiniMax H3** — use it in
**any** workflow, no All In One node required. Feed it a pass-1 A/V latent
(e.g. the **LATENT** output of ⭐ Star Minimax All In One or any MiniMax H3
sampler chain) and it runs the whole upscale-refine group internally:

1. The **video latent** is upscaled with the selected MiniMax H3 3D
   latent-upscaler model to the megapixel target (aspect ratio kept,
   32 px aligned, VAE-grid snapped).
2. The **audio latent** is carried over unchanged and the combined A/V latent
   is re-noised at the schedule's first sigma (≈0.90) — a light remix, not a
   fresh render.
3. A short **refine pass** runs, conditioned on your prompt.
4. Video and audio are decoded in-node and returned together with the
   refined latent.

```
pass-1 LATENT ──> [Star Minimax Latent Upscaler] ──> IMAGE / AUDIO / FPS / LATENT
model, clip, vae, audio_vae, prompt ──^
```

Self-contained in the StarNodes pack — the original
`Comfyui_Minimax_h3_latent_Upscaler` pack is **not** required.

For the All In One node there is also the **⭐ Star Minimax Latent Upscaler
Option** twin: same settings as an `options` bundle, and the refine pass
then reuses the AIO's pass-1 conditioning (including references) and seed.

## Required Model

```
models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors
```

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `latent` | LATENT | pass-1 MiniMax H3 A/V latent (NestedTensor with video + audio) |
| `model` | MODEL | diffusion model for the refine pass, e.g. with a turbo LoRA (`minimax_h3_fl2v_lightx2v_turbo_4step`) and/or a sage-attention patch applied |
| `clip` | CLIP | MiniMax text encoder (qwen3vl) — encodes the refine prompt |
| `vae` | VAE | MiniMax H3 video VAE |
| `audio_vae` | VAE | MiniMax H3 audio VAE |
| `audio` | AUDIO | optional — passthrough soundtrack: with the toggle on `Use 1st pass audio` this goes straight to the AUDIO output instead of the decoded pass-1 audio (e.g. the original soundtrack of a source video). Ignored on `Upscale Pass Audio` |
| **IMAGE** out | IMAGE | decoded upscaled video frames |
| **AUDIO** out | AUDIO | decoded stereo audio (pass selected by the toggle) |
| **FPS** out | FLOAT | fixed 24.0 |
| **LATENT** out | LATENT | the refined second-pass A/V latent |

## Widgets

- **prompt** — text conditioning for the refine pass (plain prompt;
  `<Picture i>` reference tags are only available through the Option twin
  inside the All In One node).
- **seed** — noise seed for the refine pass. Use the **same seed as pass 1**
  to reproduce the All In One behavior.
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
  soundtrack from the input latent (untouched) — or passes the connected
  `audio` input straight through when one is connected; `Upscale Pass Audio`
  decodes it after the refine pass, where it is lightly re-noised and
  rewritten.
- **align** (advanced) — pixel alignment of the upscaled size; keep `32` to
  avoid light banding.
- **enable_chunking** (advanced) — temporal chunking keeps VRAM in check on
  long videos; disable for short clips for full-context inference.
- **device** / **precision** (advanced) — `cuda`/`cpu`, `fp16` (default) /
  `fp32` / `bf16`. The upscaler model is parked back on CPU after inference
  so the refine pass gets the VRAM back.

## Notes

- The upscaler only upscales: pick a megapixel target **larger** than the
  pass-1 size (0.5 MP → 1.0 MP, not the other way around).
- If the input latent has no audio member, the AUDIO output carries a short
  silent placeholder.
