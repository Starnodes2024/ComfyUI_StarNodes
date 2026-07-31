# Star Latent Resize

## Description

`⭐ Star Latent Resize` lets you resize an existing latent to a **target resolution** using a
simple megapixel/preset selector and two ratio modes:

- **Keep Input Ratio** – reads the aspect ratio from the input latent and keeps it
- **Custom Ratio** – uses your own ratio string (e.g. `21:9`)

It is useful when you already have a latent (e.g. from another workflow or model) and want to
adapt its size for a new pipeline without manually calculating resolutions.

- Keeps the **input aspect ratio** by default
- Uses a megapixel / preset based size selector (default: **2 Megapixel**)
- Supports a **Custom** mode where you provide width and height directly
- Outputs the resized latent plus the final image-space width and height

## Inputs

- **LATENT** (`LATENT`, required)
  - The latent you want to resize. If nothing valid is connected, the node creates
a new empty latent at the target size.

- **ratio** (`CHOICE`, required)
  - How the aspect ratio is chosen:
    - `Keep Input Ratio` (default) – uses the aspect ratio from the input latent
    - `Custom Size` – normally used together with `resolution = custom` to rely on
      `custom_width` / `custom_height` instead of the input ratio.

- **resolution** (`CHOICE`, required)
  - Size selector, similar style to `Star Advanved Ratio/Latent`:
    - `custom`
    - `SD (≈ 512x512)`
    - `SDXL (≈ 1024x1024)`
    - `Qwen Image (≈ 1328x1328)`
    - `WAN HD (≈ 1280x720)`
    - `2 Megapixel (≈ 1408x1408)`
    - `WAN FullHD (≈ 1920x1080)`
    - `3 Megapixel (≈ 1728x1728)`
    - `4 Megapixel (≈ 2000x2000)`
    - `5 Megapixel (≈ 2240x2240)`
    - `6 Megapixel (≈ 2448x2448)`
    - `7 Megapixel (≈ 2640x2640)`
    - `8 Megapixel (≈ 2832x2832)`
    - `9 Megapixel (≈ 3008x3008)`
    - `10 Megapixel (≈ 3168x3168)`
    - `11 Megapixel (≈ 3328x3328)`
    - `12 Megapixel (≈ 3456x3456)`
    - `15 Megapixel (≈ 3888x3888)`
    - `20 Megapixel (≈ 4480x4480)`
    - `50 Megapixel (≈ 7088x7088)`
    - `100 Megapixel (≈ 10000x10000)`

- **custom_width** (`INT`, required)
  - Used when `resolution = custom`.
  - Target **image-space** width. Rounded to a multiple of 16, min 16.

- **custom_height** (`INT`, required)
  - Used when `resolution = custom`.
  - Target **image-space** height. Rounded to a multiple of 16, min 16.

## Behavior

- When `resolution` is **not** `custom`:
  - The node computes a target area from the selected preset or megapixels
    (default: **2 Megapixel (≈ 1408x1408)**).
  - If `ratio = Keep Input Ratio` and a valid latent is connected, it reads the
    spatial shape of the latent and keeps that **aspect ratio**.
  - If `ratio = Custom Ratio`, it uses the `custom_ratio` string (e.g. `21:9`).
  - Width and height are snapped to multiples of 16 (min 16) for latent compatibility.

- When `resolution = custom`:
  - The node ignores the ratio selector and uses `custom_width` and `custom_height` directly
    (snapped to multiples of 16, min 16).

- Latent resizing:
  - The node assumes a standard **8× downscale** from image to latent space.
  - For a target `width x height`, it resizes the latent tensor to
    `(height // 8, width // 8)` using bilinear interpolation.
  - If no valid latent is connected, it creates a new zero latent at the target shape.

## Outputs

- **LATENT** (`LATENT`)
  - The resized latent, with spatial dimensions matching the target resolution
    (after 8× downscale in latent space).

- **WIDTH** (`INT`)
  - Final image-space width used for the latent.

- **HEIGHT** (`INT`)
  - Final image-space height used for the latent.

## Tips

- Combine `⭐ Star Latent Resize` with your sampler or upscaler when you need to
  adapt latents between different workflows, resolutions, or models without
  hand-calculating sizes.
- Use `custom` resolution when you know the exact target size and want the
  node to just handle the latent resizing for you.
