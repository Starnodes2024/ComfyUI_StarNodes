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
  clip with audio; `image` renders one single frame (H3's native still
  convention) and decodes it with the single-frame still decode (the latent
  frame is replicated into a full 5-frame group before decoding, for far less
  banding than a bare VAE decode). Audio decoding is skipped and the audio
  VAE is not loaded (unless reference audios are connected for conditioning).
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
- **🧬 Optional RefMod injection** — connect a ⭐ Star Ref Mod Option node
  to `ref_mod_settings` to inject RefMod reference blocks from the
  ComfyUI-MiniMaxH3Mod pack into the internal conditioning, with the exact
  same retention / curve / scramble options and behavior as the *Apply H3
  RefMod* node.
- **🎬 Optional timed multiref guides** — connect a ⭐ Star Minimax Multiref
  Option node to `multiref_settings` to pin reference images/clips onto
  specific frames of the output timeline — the multi-frame / keyframe path
  of the stock template (the chained *Add Guide for MiniMax H3* nodes),
  with one start-time-in-seconds widget per connected reference.
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
| `ref_mod_settings` | REF_MOD_SETTINGS | optional — from a ⭐ Star Ref Mod Option node; appends RefMod reference blocks (ComfyUI-MiniMaxH3Mod pack, *Apply H3 RefMod* behavior) to the internal conditioning after the native references |
| `multiref_settings` | MULTIREF_SETTINGS | optional — from a ⭐ Star Minimax Multiref Option node; anchors reference images/clips at their start seconds on the output timeline (core *Add Guide for MiniMax H3* behavior) |
| `ref_image_0…8` | IMAGE | up to 9 reference images, slots expand automatically when connected |
| `ref_video_0…2` | IMAGE | up to 3 reference videos (frames @ 24 fps) |
| `ref_video_audio_0…2` | AUDIO | soundtrack paired to the same-numbered reference video |
| `ref_audio_0…2` | AUDIO | up to 3 standalone reference audios |
| **IMAGE** out | IMAGE | decoded video frames — a single still frame in `image` mode; empty when `decode_video` is off |
| **AUDIO** out | AUDIO | decoded stereo audio |
| **FPS** out | FLOAT | fixed 24.0 — connect directly to your video combine/save node |

Auto-expansion uses the same native Autogrow mechanism as the core
*MiniMax H3 Reference to Video* node — connect the last empty slot and a new
one appears.

## Widgets (defaults = template workflow)

- **mode** — **`video` (default)** renders the full clip with audio; `image`
  renders and decodes one single frame as a still image. `duration` is ignored
  in `image` mode (disabled in the UI); `aspect_ratio`, `megapixels` and
  `match_ratio_from_image` work exactly like in video mode.
- **prompt** — use `<Picture i>` / `<Video k>` / `<Audio j>` tags in connection
  order, then describe scene, motion and audio.
- **aspect_ratio** — `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`,
  **`16:9` (default)**, `2:1`, `21:9`.
- **megapixels** — dropdown with the template's size presets
  (0.2 / 0.3 / 0.4 / **0.5 default** / 0.6 / 0.7 / 0.8 / 0.9 / 0.98 / 1.0 / 1.2
  / 1.5 / 1.8 / 2.0 / **audio only**);
  0.5 MP ≈ 960×544 at 16:9, 2.0 MP ≈ 1920×1088.
  In `image` mode the extra still presets **3.0 / 4.0 / 5.0 / 6.0 / 7.0 / 8.0 MP**
  appear (single-frame stills hit their quality sweet spot from 3 MP up).
  Select **audio only** for a fixed 32×32 canvas when you only need audio output.
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
- **decode_video** — **on (default)** decodes the generated video/still with
  the video VAE. Switch it off to skip the video decode entirely: the **IMAGE**
  output stays empty and the node only delivers the **LATENT** and the
  **AUDIO** (e.g. to chain an external decode or keep VRAM free).

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
3. Pick the **mode**: `video` for clips, `image` for a high-quality single
   frame still (sized via the same aspect-ratio and megapixel widgets as
   video mode).
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
- `image` mode builds a single-frame latent (one latent frame, H3's native
  still convention) at the selected ratio and megapixel size (same presets as
  video mode), samples only that one frame, then decodes it with the
  single-frame still decode: the latent frame is replicated into a full
  5-frame group, decoded, and pixel frame 3 is kept — far less banding than a
  bare VAE decode of a lone frame. Audio decoding is skipped and the audio
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
- With a ⭐ Star Ref Mod Option connected, its ref blocks are appended after
  the native reference blocks in the internal conditioning — identical to
  running *Apply H3 RefMod* on the built conditioning — and are brought along
  (resolution-matched) into the optional upscale refine pass.
  RefMods are not addressed with `<Picture i>` tags; concat the loader's
  `prompt_hint` onto the prompt instead.
- With a ⭐ Star Minimax Multiref Option connected, its guides are anchored on
  the output timeline with the exact `MiniMaxH3AddGuide` behavior
  (`round(seconds × 24)` → frame index, stills or 17k+5 clips, negative
  seconds count from the end). Guides are timeline anchors — they are **not**
  part of the `<Picture i>` reference ordering. They work in both modes and
  are brought along (resolution-matched) into the optional upscale refine
  pass.
