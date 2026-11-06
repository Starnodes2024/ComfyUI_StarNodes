# ⭐ Star Minimax H3 Image Inpainter

## Overview

**Star Minimax H3 Image Inpainter** is an all-in-one inpainting node for the
**MiniMax H3** model — the ⭐ Star Flux2/Qwen-Image-Edit Inpainter workflow
rebuilt for H3's native **image mode**. It replicates a complete inpaint
pipeline in a single node: smart crop-and-stitch around the masked area,
Qwen3-VL + reference conditioning, the model's own per-token denoise-mask
inpainting, sampling on the packed AV latent and the single-frame still decode.

Unlike the Flux2/Qwen node it needs no concat-conditioning tricks — H3
consumes a denoise mask natively. Each DiT token row runs at `mask × sigma`,
preserved rows are pinned to the clean source latent and re-injected every
step, so a soft/blurred mask edge translates directly into per-row denoise
strength (the model's built-in differential diffusion).

## Required Models

```
models/diffusion_models/  minimax_h3_ref2va_*.safetensors   (any H3 diffusion model)
models/text_encoders/     qwen3vl_32b_minimax_h3_*.safetensors  (CLIP type: minimax)
models/vae/               minimax_h3_video_vae_*.safetensors    (the video VAE)
```

The audio VAE is **not** needed — image mode works on a single still frame.

## Inputs

### Required
- **model**: The MiniMax H3 diffusion model (from UNETLoader, or the `MODEL`
  output of a ⭐ Star Minimax All In One node)
- **clip**: The MiniMax H3 text encoder (Qwen3-VL-32B, loaded with clip type `minimax`)
- **vae**: The MiniMax H3 **video** VAE
- **image**: The source image to be inpainted
- **mask**: The mask marking the areas to regenerate (white = inpaint)
- **prompt**: What should appear in the masked area. The source crop is
  `<Picture 1>`, extra reference images are `<Picture 2>`, `<Picture 3>`, …
- **reference**: How the source crop enters the conditioning as `<Picture 1>`
  - `masked source` (default) — shows the crop with the inpaint area blanked
    to gray, the hole signals what to fill (classic inpaint convention)
  - `source` — shows it untouched, edit style; the model can see what was
    there (good for "change X into Y" prompts)
  - `off` — no picture reaches the text encoder at all (text only; the
    denoise mask still anchors the unmasked area)
- **megapixels**: Render resolution of the inpaint crop. `native` (default)
  keeps the crop's own pixels; the MP presets (0.5 – 8.0) rescale crop + mask
  to that pixel area first — H3 stills hit their quality sweet spot around
  3 MP and up. The result is scaled back to the crop size for stitching.
- **seed**, **steps** (20), **sampler_name** (`res_multistep`),
  **scheduler** (`simple`), **denoise** (1.0)
- **cfg**: CFG scale (default 1.0 — the positive-only basic guider, the H3
  preset). Set it higher to activate the negative prompt.
- **negative_prompt**: Only encoded when `cfg` differs from 1.0.

### Advanced
- **expand_pixels**: Context margin around the masked area (default 32)
- **mask_blur**: Grow + blur of the denoise mask (default 16) — soft edges
  become per-row denoise strength on H3
- **blend_pixels**: Feather width for the stitch back into the original
  (default 16)
- **ref_image_size**: `match` (default, refs scaled down to the render area)
  or `max` (2048 px short edge, slower, best identity fidelity)

### Connectors
- **ref_image_…**: Extra reference images, autogrowing slots — they become
  `<Picture 2>`, `<Picture 3>`, … in connection order
- **sigmas**: Optional exact sigma schedule (e.g. from a ⭐ Star Custom
  Sigmas node) — overrides `scheduler`, `steps` and `denoise`
- **preview**: Optional ⭐ Star Preview node for a live sampling preview

## Outputs
- **IMAGE**: The original image with the inpainted region stitched back in
- **LATENT**: The sampled packed AV latent (NestedTensor video+audio)
- **MASK**: The original mask (passed through)
- **seed**: The seed used for generation

## Usage

1. Connect the H3 diffusion model, the minimax text encoder and the H3 video
   VAE (e.g. straight from a ⭐ Star Minimax All In One node's `MODEL` /
   `CLIP` / `VAE` outputs).
2. Provide an image and a mask (white = inpaint).
3. Write a prompt for the fill — refer to the source as `<Picture 1>` and to
   extra references as `<Picture 2>`, `<Picture 3>`, …
4. (Optional) Raise `megapixels` for a higher-quality render, connect extra
   reference images, tune `mask_blur` / `blend_pixels`.
5. Run — the context area is cropped, sampled with the native denoise mask
   and stitched back automatically.

## Features

### Native H3 inpainting
- Uses the model's own per-token denoise mask — no `InpaintModelConditioning`,
  no Differential Diffusion patch; soft masks are natively per-row sigma
- Preserved regions are pinned to the source latent at the cond timestep and
  re-injected every sampling step — the surroundings stay untouched even
  inside the rendered crop
- The audio member of the packed latent is pinned to silence

### Image-mode pipeline
- Single-frame still latent (`[1, 24, 1, H/16, W/16]` video member + the
  silent audio member H3's packed layout always carries), identical to the
  `image` mode of ⭐ Star Minimax All In One
- Decoded with the single-frame still decode (latent replicated into a
  5-frame group, pixel frame 3 kept) — far less banding than a bare VAE decode

### Shared crop & stitch
- Same battle-tested crop-and-stitch implementation as the ⭐ Star
  Flux2/Qwen-Image-Edit Inpainter: holes filled, mask grown and blurred,
  context margin added, seamless feathered blend back into the original

## Notes

- One image at a time — MiniMax H3 supports batch size 1.
- `denoise < 1.0` starts sampling mid-schedule for subtler edits; 1.0 is the
  normal full-repaint value.
- `'beta'` or `'normal'` schedulers tend to outperform `'simple'` for
  reference-heavy prompts.
- `masked source` is the safer default for object removal; `source` can work
  better when the prompt edits (rather than removes) what is under the mask.
- If the crop is small, a `megapixels` preset of 2.0–4.0 usually gives visibly
  better fills than `native`.
