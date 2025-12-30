# ⭐ Star Upscale Refiner 2 [BETA]

## Overview

`StarUpscaleRefiner2` is an all-in-one upscaling + refinement node that takes direct `MODEL` + `CLIP` inputs and supports selecting a *separate decode VAE* via dropdown.

This node is **experimental**.

## Inputs

### Required

- `model` (`MODEL`)
- `clip` (`CLIP`)
- `vae` (`VAE`)
  - Used for **encoding** the input image into latents.
- `vae_decode_name`
  - `same_as_encode` (uses the connected `vae`)
  - or select a VAE from your `models/vae` folder

### Optional

- `IMAGE` (`IMAGE`)
- `controlnet_name`
- `UPSCALE_IMAGE`
- `UPSCALE_MODEL`
- `OUTPUT_LONGEST_SIDE` (display name: `Output Size (longest)`)

## 2x VAE behavior (WanVAE / 12-channel decode)

Some VAEs (commonly Wan 2.x “2x VAE” decode models) output more than 3 channels and produce a **2x larger image** on decode.

To keep the **final decoded output size** aligned with what you request, the node automatically adjusts the internal render size:

- If the selected decode VAE has `output_channels > 3` (e.g. 12), the node will use:
  - `requested_longest_side = OUTPUT_LONGEST_SIDE / 2`

Example:

- `Output Size (longest) = 4000`
- 2x decode VAE selected
- internal sampling/resize target becomes `2000`
- final decoded output becomes ~`4000`

## Notes

- The decode output is post-processed to handle 2x VAEs (pixel-shuffle style behavior) directly inside StarNodes (no external dependencies).
