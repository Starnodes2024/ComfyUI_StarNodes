# ⭐ Star Qwen Image Edit Inputs

Stitch up to 4 images into a single canvas and generate a Qwen-sized latent. Designed for quick image collages and resolution-aligned editing workflows.

- Background color: white
- Output image size rules:
  - 1 image: keep original size; if larger, scale longest side to 1328
  - 2 images: 1×2 horizontal stitch; if taller than 1328, scale height to 1328
  - 3 images: 2×2 grid; missing fourth cell is white; final 1328×1328
  - 4 images: 2×2 grid; final 1328×1328

All latent sizes are multiples of 16 (spatial) and 8 (latent grid factor).

## Inputs

- __image1__ (required)
  - `IMAGE`. The first image. Used for stitching and for the "Use Best Ratio from Image 1" selection.

- __image2__, __image3__, __image4__ (optional)
  - `IMAGE`. Additional images to stitch.

- __qwen_resolution__
  - Dropdown options:
    - `Use Best Ratio from Image 1`
    - `1:1 (1328x1328)`
    - `16:9 (1664x928)`
    - `9:16 (928x1664)`
    - `4:3 (1472x1104)`
    - `3:4 (1104x1472)`
    - `3:2 (1584x1056)`
    - `2:3 (1056x1584)`
    - `5:7 (1120x1568)`
    - `7:5 (1568x1120)`
    - `Free Ratio (custom)`

- __batch_size__
  - Integer. Number of latent samples to output.

- __custom_width__, __custom_height__
  - Used when selecting `Free Ratio (custom)`.
  - Values are clamped internally to multiples of 16.

## Outputs

- __stitched__ (IMAGE)
  - The stitched single-frame image tensor `[1, H, W, C]`.

- __latent__ (LATENT)
  - Zero-initialized latent dict `{ "samples": tensor }` with shape `[batch, 4, H/8, W/8]`.

- __width__ (INT)
- __height__ (INT)

## Behavior Details

- 1 image:
  - If largest side > 1328, downscale so the longest side is 1328; otherwise keep original size.
- 2 images:
  - Both are resized to the same height (the larger of the two), concatenated horizontally, then if height > 1328, downscale by height to 1328.
- 3/4 images:
  - Each image is letterboxed into a 664×664 cell (white background) to preserve individual aspect ratios.
  - 2×2 grid assembly yields exactly 1328×1328.

## Notes

- "Use Best Ratio from Image 1" picks the Qwen preset whose aspect ratio is closest to `image1`.
- Latent spatial sizes are enforced to multiples of 16; the latent tensor uses an 8× downsample factor.
- The stitched image is not forcibly resized to the latent size; it follows the rules above.
