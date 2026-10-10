# ⭐ Star Custom Sigmas

## Overview

**Star Custom Sigmas** turns a comma-separated sigma list into a real `SIGMAS` output, so the connected sampler runs with **exactly** the sigma values you pick — no scheduler math, no step interpolation.

Pick one of the baked flow-match presets — **8**, **10**, **12** or **16 Steps** — or choose **Custom** to run the text field as-is. Selecting a preset also writes its values into the text field, so you can use any preset as a starting point for your own schedule (editing the text switches the dropdown to **Custom**).

N values = N-1 sampling steps. The default 10-step preset:

```
1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.870114, 0.845148, 0.704534, 0.559551, 0.414568, 0.0
```

## Inputs

### Required

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **preset** | COMBO | `10 Steps` | Baked sigma schedule: `8` / `10` / `12` / `16 Steps`, or `Custom` to use the text field. |
| **sigmas** | STRING | `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.870114, 0.845148, 0.704534, 0.559551, 0.414568, 0.0` | Comma-separated sigma values, high to low — only used when preset is `Custom`. Multiline input — newlines and extra spaces are fine. End with `0.0` for a full denoise. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| **sigmas** | SIGMAS | The parsed sigma list as a tensor. |

## Usage

```
┌─────────────────────────┐          ┌──────────────────────────┐
│ ⭐ Star Custom Sigmas    │─ sigmas ─►│ options                  │
└─────────────────────────┘          │ ⭐ StarSampler (Unified)  │
                                     └──────────────────────────┘
```

1. Add `⭐ Star Custom Sigmas`.
2. Edit the text field if needed (the default already gives the 10-step schedule).
3. Connect `sigmas` → `options` on `⭐ StarSampler (Unified)` — also works on the `options` input of `⭐ Star Qwen2 Outpainter` and `⭐ Star Flux2/Qwen-Image-Edit Inpainter`.

While connected, the sampler **overrides its `steps`, `scheduler` and `use_ltx_scheduler` widgets** and runs exactly the listed sigmas — the default is 10 steps ending at 0 (full denoise). The `sampler_name` widget still applies (e.g. `euler`), and StarSampler's `info` output reports the scheduler as `custom sigmas`.

Because the output is a real `SIGMAS` type, the same node also plugs into **vanilla ComfyUI sigmas inputs** — e.g. `SamplerCustomAdvanced`, `SplitSigmas`, `FlipSigmas` — so it works outside the StarNodes sampler too.

## Notes

- **Sigmas are used verbatim** — no rescaling to the model's sigma range. The default `1.0 → 0.0` list fits flow models (Flux, Qwen, Z-Image, …). Classic SD/SDXL models expect larger sigmas (e.g. `14.6, …, 0.0`) — enter a matching schedule for those.
- To keep residual noise (partial denoise), simply end the list above `0.0`.
- Ignored — with a log warning — while a `detail_schedule` is connected to StarSampler (the detail-daemon path builds its own sigma handling).
- Fewer than 2 parseable values falls back to `1.0, 0.0` with a warning.

**Category**: ⭐StarNodes/Sampler

**Node Name**: StarCustomSigmas

**Display Name**: ⭐ Star Custom Sigmas
