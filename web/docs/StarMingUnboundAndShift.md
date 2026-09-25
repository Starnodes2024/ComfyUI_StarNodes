# ⭐ Star Ming Unbound + Shift

Combines the **Star Krea2 Unbound** prompt-adherence enhancer with a **model sampling shift** patch, tuned for **Ming-Image**.

- **Category**: `⭐StarNodes/Sampler`
- **Node name**: `StarMingUnboundAndShift`
- **Output**: `model` (`MODEL`)

## Inputs

- **model** (`MODEL`)
  - The diffusion model to patch (e.g. the Ming-Image DiT from a model loader).

- **shift** (`FLOAT`, default `3.16`)
  - Flow-matching sampling shift applied to the model.
  - Ming-Image ships with `3.16`, the reference dynamic-shift value for the 1024 bucket.
  - Higher values concentrate the schedule on high-noise steps (more structure/detail shaping); lower values spread steps more evenly.
  - The model's own multiplier and noise scale are preserved - only the shift changes.

## Notes

- The Unbound enhancer only activates on models with the Krea2-style `txtfusion` text stack; on Ming-Image it passes through cleanly.
- Works with any flow-matching model that uses discrete-flow sampling (Ming-Image, Z-Image, AuraFlow-style, etc.).
- Place it between the model loader and the sampler.
