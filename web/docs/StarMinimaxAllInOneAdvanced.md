# ⭐ Star Minimax All In One Advanced

## Overview

**Star Minimax All In One Advanced** is the complete ⭐ Star Minimax All In One
pipeline plus three optional injection connectors, so conditioning and latents
built by **external nodes** — e.g. the **ComfyUI-BFSNodes** shot nodes
(`BFSShotH3ConditioningSimple` and friends) or any other node that outputs
`CONDITIONING` / `LATENT` — can be sampled with this node's sampler, sigmas,
upscale pass and decode stages.

Everything the base node does still works: model loading, the internal
`<Picture i>` / `<Video k>` / `<Audio j>` reference conditioning, resolution
presets, RefMod and multiref option injection, the second-pass latent upscale,
sound enrichment and video/audio VAE decoding. The extra inputs simply let you
replace parts of the internal pipeline with connected values.

## The three injection connectors

| Connector | Type | Effect when connected |
|---|---|---|
| `positive` | CONDITIONING | The internal prompt + reference encode is **skipped entirely** — the text encoder is not even loaded, and `prompt`, `clip_*` widgets and all `ref_*` inputs are ignored. The connected conditioning is sampled instead. |
| `negative` | CONDITIONING | Sampling switches from the basic positive-only guider to a real `CFGGuider` running at the `cfg` widget value (at cfg 1.0 the negative is ignored by the sampler anyway). |
| `latent` | LATENT | Sampling runs on the connected latent. `aspect_ratio`, `megapixels` and `duration` are ignored — canvas size and frame count are read back from the latent itself. |

Any combination works:

- **positive only** — the internal empty AV latent is still built from the
  aspect ratio / megapixels / duration widgets; only the encode is replaced.
- **latent only** — the internal ref2va conditioning still runs (prompt and
  references are honored); only the empty latent is replaced.
- **positive + latent** — a full external conditioning bundle (the BFS shot
  workflow): only model loading, sampling and decode run inside the node.
- **negative only** — pairs with either internal or external positive
  conditioning and activates the `cfg` widget.
- **nothing connected** — behaves exactly like ⭐ Star Minimax All In One.

The `model_override` and `vae_override` connectors behave exactly as on the
base node — so a model patched by an upstream node (e.g. the `model` output of
a BFS conditioning node with a temporal backend) can simply be wired into
`model_override`.

## BFSNodes shot workflow

The typical wiring for the BFS shot-planner workflows:

```
BFSShotH3ConditioningSimple
    positive ──────────────► positive   ⭐ Star Minimax All In One Advanced
    latent   ──────────────► latent         (seed / steps / sampler /
    model    ──────────────► model_override scheduler / sigmas / decode)
```

This gives you the BFS shot conditioning (guide-aligned references, per-shot
prompts, segmentation helpers) while keeping the Star node's sampler controls,
custom-sigmas support, second-pass latent upscale, sound enrichment and the
internal VAE decode — without rebuilding the sampling chain by hand.

## Extra widget

- **cfg** (advanced, default `1.0`) — CFG scale for the guider. Only used when
  a `negative` conditioning is connected; at `1.0` the negative is ignored by
  the sampler anyway. A warning is logged when `cfg ≠ 1.0` is set without a
  negative.

## Everything else is identical to the base node

- All widgets, option connectors (`sound_settings`, `sigmas`, `options`,
  `ref_mod_settings`, `multiref_settings`, `preview`) and the autogrowing
  `ref_*` inputs behave exactly as documented in
  [StarMinimaxAllInOne](StarMinimaxAllInOne.md).
- RefMod blocks and multiref guides are applied on top of **whichever**
  conditioning is active — including a connected external `positive`.
- The upscale refine pass uses the same conditioning (resolution-matched) and
  also honors a connected `negative` + `cfg`.
- Outputs are unchanged: `IMAGE`, `AUDIO`, `FPS`, `LATENT`, `MODEL`, `CLIP`,
  `VAE`, `AUDIO_VAE`. When `positive` is connected the internal text encoder
  is not loaded and the `CLIP` output is `None` — don't wire it anywhere in
  that setup.
- A connected latent without an audio member yields the same short silent
  placeholder as `image` mode on the `AUDIO` output.

## Notes

- Reference inputs connected while `positive` is linked are ignored (the node
  logs a warning) — external conditioning already contains its own reference
  blocks; connect the matching latent too if the conditioning was built for a
  specific canvas/duration.
- `length` of an external latent is recovered from its video-member shape
  (the 17k+5 frame grid inverse) and used for logging and ref-video capping.
- Requires the same models/frontend as the base node — see
  [StarMinimaxAllInOne](StarMinimaxAllInOne.md#required-models).
