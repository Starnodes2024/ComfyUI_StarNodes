# ⭐ Star LTX Scheduler (Qwen Image 2.x)

## Overview

The **Star LTX Scheduler (Qwen Image 2.x)** is an options node that replicates the official **Qwen-Image-2.1 dynamic shifting schedule** inside ComfyUI. It works with **⭐ StarSampler (Unified)**, **⭐ Star Qwen2 Outpainter** and **⭐ Star Flux2 Inpainter** via their `options` input.

It fixes a known generation issue: at resolutions above 1024×1024 (e.g. 2048×2048), Qwen Image 2.1 can produce noisy or grid-patterned output because the default model sampling uses a **fixed shift** tuned for 1024×1024 (mu ≈ 0.69). The official pipeline instead scales the shift with the image **token count**, so larger images spend more effort on the high-noise steps.

This node does not output `SIGMAS` for a generic sampler — it emits a `STARNODES_OPTIONS` bundle that the Star sampler nodes understand. When connected, the sampler builds the sigma curve internally and ignores its own `scheduler` and `steps` widgets.

> **Easier alternative:** the same schedule is available without this node — all three sampler nodes have a **`use_ltx_scheduler`** toggle (default: on) that acts exactly as if this node were connected with its defaults: **40 steps** and the token count from the sampler's own latent. This node remains for explicit step-count control.

## Inputs

### Required

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **steps** | INT | 40 | Number of sampling steps (official Qwen-Image-2.1 default). Overrides the `steps` widget on the sampler. |

### Optional

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **latent** | LATENT | None | The latent being sampled. Its spatial size defines the image token count (Qwen Image 2.x uses one token per latent pixel: 64×64 = 4096 tokens at 1024px). If left unconnected, the sampler derives the token count from its own latent — required anyway for the Qwen out-/inpainter nodes whose latent is built internally. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| **options** | STARNODES_OPTIONS | Options bundle for the `options` input of ⭐ StarSampler (Unified), ⭐ Star Qwen2 Outpainter and ⭐ Star Flux2 Inpainter. |

## How It Works

The node ships the official Qwen-Image-2.1 scheduler parameters to the sampler:

- `base_shift` 0.5 @ 256 tokens, `max_shift` 0.9 @ 8192 tokens
- `shift_terminal` 0.02 — the smallest sigma is stretched to this value
- Token count is read from the connected latent, so the schedule always matches the actual resolution

For each step the sigma is warped with the exponential time shift

```
mu    = tokens * (max_shift - base_shift) / (8192 - 256) + base_shift - 0.5-term
sigma = e^mu / (e^mu + (1/t - 1))
```

and the tail is stretched so the last non-zero sigma equals `terminal` (0.02), followed by a final 0 sigma for full denoise. A `denoise` below 1.0 on the sampler shortens the schedule like the stock BasicScheduler does.

## Usage

```
Empty Latent (e.g. 2048x2048) ──► latent (optional - else the sampler's own latent decides)
                                   ┌──────────────────────────────┐
                                   │ Star LTX Scheduler (Qwen 2.x)│
                                   └──────────────┬───────────────┘
                                                  │ options
┌──────────────────────────────────────┐          ▼
│ ⭐ StarSampler (Unified)             │◄── options
│ ⭐ Star Qwen2 Outpainter             │◄── options
│ ⭐ Star Flux2 Inpainter              │◄── options
└──────────────────────────────────────┘
```

- Sampler: `euler` (the schedule only defines the sigma curve — any sampler works)
- CFG/guidance: keep your normal Qwen setting (official txt2img uses ~1.0)
- If the model is not a Flux-style flow model, the options are ignored and the sampler logs a warning.

## Notes

- Designed for **Qwen Image 2.x** (text-to-image and edit). Other flow models can technically consume the schedule, but the anchors are the official Qwen values.
- Works together with the `preview` input and the standard seed/denoise controls of the Star sampler nodes.
- In ⭐ StarSampler (Unified) the options are ignored when a `detail_schedule` is connected — the detail-daemon path runs its own sigma handling.
- Token count = latent pixels (H×W of the latent), which is Qwen Image 2.x's one-token-per-latent-pixel convention.

**Category**: ⭐StarNodes/Sampler

**Node Name**: StarNodes_LTXScheduler_Qwen2x_Options

**Display Name**: ⭐ Star LTX Scheduler (Qwen Image 2.x)
