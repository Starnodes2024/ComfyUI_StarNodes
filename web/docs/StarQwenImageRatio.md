# Star Qwen Image Ratio

## Description
Dropdown-driven aspect ratio selector that outputs an empty SD3-compatible latent plus width and height integers.

## Category
⭐StarNodes/Image And Latent

## Inputs
- ratio: One of the preset ratios or "Free Ratio (custom)"
- batch_size (INT, default 1): Latent batch dimension
- custom_width (INT): Used when selecting Free Ratio (divisible by 16 enforced)
- custom_height (INT): Used when selecting Free Ratio (divisible by 16 enforced)

## Outputs
- latent (LATENT): Dict with `samples` tensor sized to the selected resolution (downsampled by 8)
- width (INT)
- height (INT)

## Supported Ratios
- 1:1 → 1328 × 1328
- 16:9 → 1664 × 928
- 9:16 → 928 × 1664
- 4:3 → 1472 × 1104
- 3:4 → 1104 × 1472
- 3:2 → 1584 × 1056
- 2:3 → 1056 × 1584
- 5:7 → 1120 × 1568
- 7:5 → 1568 × 1120

## Usage
1. Choose a ratio from the dropdown
2. If using Free Ratio, set custom width/height (multiples of 16 recommended)
3. Connect `latent` to SD3/SD3.5 workflows expecting a LATENT dict
4. Use `width` and `height` to configure downstream nodes

## Notes
- Width/height are enforced to be divisible by 16, and latent size is aligned to /8 downsampling
- Increase `batch_size` to create batched latents
