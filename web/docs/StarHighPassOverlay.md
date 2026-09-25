# ⭐ Star HighPass Overlay

Photoshop-style **High Pass + Overlay** sharpening filter.

It replicates the classic Photoshop technique:

1. The input image is blurred with a Gaussian kernel.
2. The residual (`image - blur`) is placed on a neutral grey layer (the "High Pass" layer).
3. That layer is blended back in **Overlay** mode, so only edge/detail contrast changes.
4. The result is flattened and output.

- **Category**: `⭐StarNodes/Image And Latent`
- **Node name**: `StarHighPassOverlay`
- **Output**: `image` (`IMAGE`, always RGB)

## Inputs

- **image** (`IMAGE`)
  - RGB or RGBA input. RGBA images are converted to RGB (alpha channel dropped).

- **radius** (`INT`, default `10`, max `250`)
  - Radius of the high pass filter - controls which detail scale is boosted.
  - Small values sharpen fine texture, large values boost broader local contrast.

- **strength** (`FLOAT`, default `1.0`, range `0.0`-`1.0`)
  - Opacity of the high pass overlay layer, like the Photoshop layer percentage.
  - `0.0` = original image, `1.0` = full overlay effect.

## Notes

- Edge-safe: the blur uses reflect padding, so no dark borders appear.
- Flat/grey regions are untouched - Overlay at 50% grey is a no-op.
