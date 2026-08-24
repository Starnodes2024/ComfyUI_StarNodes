# ⭐ Star Minimax All In One

## Overview

**Star Minimax All In One** is a single ComfyUI node that runs the complete
**MiniMax H3 reference-to-video** pipeline in-process — no group wrappers, no
sub-workflows. Everything from the stock template workflow's *Models /
Conditioning / Sampling / Decoding* groups lives inside one node, with only the
user-facing connectors exposed.

It replicates (in-process) the exact behavior of:

`UNETLoader` → `CLIPLoader` (type `minimax`) → `VAELoader` (video) →
`VAELoaderKJ`-style audio VAE load with FP32 precision → `ResolutionSelector` →
duration math → `MiniMaxH3ReferenceToVideo` conditioning → `RandomNoise` →
`BasicGuider` → `KSamplerSelect` → `BasicScheduler` → `SamplerCustomAdvanced` →
`VAEDecode` → `VAEDecodeAudio`.

## Key Features

- **🎬 Single-node pipeline** — model loading, reference conditioning, sampling
  and video/audio VAE decoding all run inside the node, no sub-graph needed.
- **🖼️📽️ Image / Video mode selector** — `video` (default) renders the full
  clip with audio; `image` renders exactly 9 frames and outputs only frame
  index 8 as a still image. The last frame carries the best quality. Audio
  decoding is skipped and the audio VAE is not loaded (unless reference audios
  are connected for conditioning).
- **🧩 Reference inputs work in both modes** — `image` mode accepts the same
  reference images / videos / audios as `video` mode (ideal for image edits:
  connect the source image as `ref_image_0` and prompt with `<Picture 1>`).
- **🖼️ Up to 9 reference images, 3 reference videos, 3 standalone audios** —
  reference image/video/audio slots grow automatically through the same native
  Autogrow mechanism the core MiniMax H3 node uses.
- **🔊 Audio + video output** — decodes both the video frames and the stereo
  soundtrack in one go.
- **📐 Resolution presets** — aspect ratio + megapixel selector with optional
  ratio matching from the first reference image (same math as the core
  `ResolutionSelector`).
- **🎚️ Audio VAE precision selectable** — `fp32` (default, KJ-loader preset),
  `fp16` or `bf16`.
- **🔌 Optional MODEL override** — connect an external model (e.g. a
  sage-attention-patched MiniMax H3) and the internal `diffusion_model`
  dropdown is ignored.
- **🎚️ Optional sound enrichment** — connect a ⭐ Star Video Sound Enricher
  Option node to `sound_settings` and the soundtrack is cleaned up and
  enriched internally (de-harsh, bass/warmth boost, high-fizz taming),
  delivered at 44.1 kHz or the source rate, never downsampled.
- **🔍 Optional second-pass latent upscale** — connect a ⭐ Star Minimax
  Latent Upscaler Option node to `options`: the pass-1 video latent is
  upscaled with a MiniMax H3 3D latent-upscaler model and refined in a short
  second sampling pass (baked 3/4/5-step schedules, same conditioning and
  same seed — references are resolution-matched automatically). The audio
  toggle on the option node picks which pass the audio output is decoded
  from.
- **📊 Live readout + animated progress bar** — a readout line under the widgets
  shows the resolved `width × height • MP • frames`; an animated DOM progress
  bar appears during execution (indeterminate shimmer while models load and
  references encode, then per-step percentage during sampling, finishing with
  the decode phase).

## Required Models

A ComfyUI version with MiniMax H3 support (`comfy_extras.nodes_minimax_h3`) and
a frontend with Autogrow input support — the same requirements as the stock
MiniMax H3 template workflow.

```
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
```

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `model_override` | MODEL | optional — when connected, the internal `diffusion_model` dropdown is ignored. Use it for sage-attention-patched or otherwise modified models. |
| `sound_settings` | SOUND_SETTINGS | optional — from a ⭐ Star Video Sound Enricher Option node; the generated soundtrack is processed with these settings before it leaves the node. Ignored in `image` mode without audio |
| `options` | UPSCALE_SETTINGS | optional — from a ⭐ Star Minimax Latent Upscaler Option node; runs a second-pass latent upscale + refine. Ignored when `megapixels` is `audio only` |
| `ref_image_0…8` | IMAGE | up to 9 reference images, slots expand automatically when connected |
| `ref_video_0…2` | IMAGE | up to 3 reference videos (frames @ 24 fps) |
| `ref_video_audio_0…2` | AUDIO | soundtrack paired to the same-numbered reference video |
| `ref_audio_0…2` | AUDIO | up to 3 standalone reference audios |
| **IMAGE** out | IMAGE | decoded video frames — a single still (frame index 8) in `image` mode |
| **AUDIO** out | AUDIO | decoded stereo audio |
| **FPS** out | FLOAT | fixed 24.0 — connect directly to your video combine/save node |

Auto-expansion uses the same native Autogrow mechanism as the core
*MiniMax H3 Reference to Video* node — connect the last empty slot and a new
one appears.

## Widgets (defaults = template workflow)

- **mode** — **`video` (default)** renders the full clip with audio; `image`
  renders 9 frames and outputs only frame index 8 as a still image (the
  best-quality frame). `duration` is ignored in `image` mode (disabled in the
  UI); `aspect_ratio`, `megapixels` and `match_ratio_from_image` work exactly
  like in video mode.
- **prompt** — use `<Picture i>` / `<Video k>` / `<Audio j>` tags in connection
  order, then describe scene, motion and audio.
- **aspect_ratio** — `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`,
  **`16:9` (default)**, `2:1`, `21:9`.
- **megapixels** — dropdown with the template's size presets
  (0.2 / 0.3 / 0.4 / **0.5 default** / 0.6 / 0.7 / 0.8 / 0.9 / 0.98 / 1.0 / 1.2
  / 1.5 / 1.8 / 2.0 / **audio only**);
  0.5 MP ≈ 960×544 at 16:9, 2.0 MP ≈ 1920×1088.
  Select **audio only** for a fixed 32×32 canvas when you only need audio output.
  Same presets in both modes.
- **match_ratio_from_image** — when ON and a reference image is connected, the
  closest matching ratio of the first reference image is picked at the
  selected pixel size.
- **duration** — seconds @ 24 fps, snapped internally to the 17k+5 frame grid
  (5 s → 124 frames), same formula as the template's Math Expression node.
  Only used in `video` mode (disabled in the UI when `image` is selected).
- **ref_image_size** — `match` (default) / `max`.
- **seed** (randomize / fixed / increment / decrement), **steps** 20,
  **sampler** `res_multistep`, **scheduler** `simple`, **denoise** 1.0.
- **diffusion_model / weight_dtype / clip_name / clip_type / clip_device**.
- **vae_name** (video VAE).
- **audio_vae_name**, **audio_vae_precision** (`fp32` default — selectable
  fp32/fp16/bf16), **audio_vae_device**.

## In-node UI

- A readout line under the widgets shows the resolved
  `width × height • MP • frames` and updates live as you change ratio / MP /
  duration.
- An animated progress bar appears inside the node during execution:
  indeterminate shimmer while models load and references encode, then
  per-step percentage during sampling, finishing with the decode phase.

## Usage

1. Make sure the four MiniMax H3 model files listed above are present.
2. Add the node: **⭐StarNodes/Video → ⭐ Star Minimax All In One**.
3. Pick the **mode**: `video` for clips, `image` for a high-quality still
   (frame index 8 of a fully rendered 9-frame run, sized via the same
   aspect-ratio and megapixel widgets as video mode).
4. (Optional) Connect reference images, reference videos (with paired audio)
   and/or standalone reference audios to the autogrowing slots.
5. Write your prompt using `<Picture i>` / `<Video k>` / `<Audio j>` tags in
   connection order, then describe the scene, motion and audio you want.
6. Pick aspect ratio, megapixels and (video mode) duration.
7. Connect the **IMAGE**, **AUDIO** and **FPS** outputs straight into your
   video combine / save node (e.g. ⭐ Star Video Compressor). In `image` mode
   wire **IMAGE** into a Save/Preview Image node — the **AUDIO** output carries
   a short silent placeholder.

## Notes

- The internal pipeline is identical in logic to the stock nodes — no behavior
  is changed, only the wiring is collapsed into one node.
- `image` mode builds a video latent with exactly 9 temporal frames at the
  selected ratio and megapixel size (same presets as video mode), samples and
  VAE-decodes all 9 frames, then returns frame index 8 as the still. Audio decoding is skipped and the audio
  VAE is not loaded unless reference audios are connected.
- For image edits in `image` mode, connect your source image to `ref_image_0`
  (and more references if needed) and reference them in the prompt with
  `<Picture 1>` etc. — exactly like in `video` mode.
- `'beta'` or `'normal'` schedulers tend to outperform `'simple'` for
  reference-heavy prompts.
- Use the `model_override` input when you want to feed in a MiniMax H3 model
  pre-patched with sage/flash attention — the internal dropdown and
  `weight_dtype` are then ignored.
- With an upscale options node connected, the **IMAGE** and **LATENT** outputs
  come from the refined second pass, **AUDIO** comes from the pass selected by
  the option node's audio toggle (default: pass 1), and **MODEL** remains the
  pass-1 model.
