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
| `ref_image_0…8` | IMAGE | up to 9 reference images, slots expand automatically when connected |
| `ref_video_0…2` | IMAGE | up to 3 reference videos (frames @ 24 fps) |
| `ref_video_audio_0…2` | AUDIO | soundtrack paired to the same-numbered reference video |
| `ref_audio_0…2` | AUDIO | up to 3 standalone reference audios |
| **IMAGE** out | IMAGE | decoded video frames |
| **AUDIO** out | AUDIO | decoded stereo audio |
| **FPS** out | FLOAT | fixed 24.0 — connect directly to your video combine/save node |

Auto-expansion uses the same native Autogrow mechanism as the core
*MiniMax H3 Reference to Video* node — connect the last empty slot and a new
one appears.

## Widgets (defaults = template workflow)

- **prompt** — use `<Picture i>` / `<Video k>` / `<Audio j>` tags in connection
  order, then describe scene, motion and audio.
- **aspect_ratio** — `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`,
  **`16:9` (default)**, `21:9`.
- **megapixels** — dropdown with the template's size presets
  (0.2 / 0.3 / 0.4 / **0.5 default** / 0.6 / 0.7 / 0.8 / 0.9 / 0.98 / 1.0 / 1.2
  / 1.5 / 1.8 / 2.0);
  0.5 MP ≈ 960×544 at 16:9, 2.0 MP ≈ 1920×1088.
- **match_ratio_from_image** — when ON and a reference image is connected, the
  closest matching ratio of the first reference image is picked at the
  selected pixel size.
- **duration** — seconds @ 24 fps, snapped internally to the 17k+5 frame grid
  (5 s → 124 frames), same formula as the template's Math Expression node.
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
3. (Optional) Connect reference images, reference videos (with paired audio)
   and/or standalone reference audios to the autogrowing slots.
4. Write your prompt using `<Picture i>` / `<Video k>` / `<Audio j>` tags in
   connection order, then describe the scene, motion and audio you want.
5. Pick aspect ratio, megapixels and duration.
6. Connect the **IMAGE**, **AUDIO** and **FPS** outputs straight into your
   video combine / save node (e.g. ⭐ Star Video Compressor).

## Notes

- The internal pipeline is identical in logic to the stock nodes — no behavior
  is changed, only the wiring is collapsed into one node.
- `'beta'` or `'normal'` schedulers tend to outperform `'simple'` for
  reference-heavy prompts.
- Use the `model_override` input when you want to feed in a MiniMax H3 model
  pre-patched with sage/flash attention — the internal dropdown and
  `weight_dtype` are then ignored.
