# ⭐ Star Special Filters

## Description
Multi-purpose image filter node. The first available filter simulates a Photoshop-style High Pass sharpening pass: it creates a high-pass version of the image (original minus a blurred copy) and adds it back to the original, enhancing edges and fine details.

## Inputs
- **image** (IMAGE, required): Source image.
- **radius** (INT, required): Blur radius used to create the low-frequency (blurred) image. Larger radius targets broader features. Default: 3.
- **strength** (FLOAT, required): Amount of high-pass detail added back to the original image. Default: 1.0.

## Outputs
- **image** (IMAGE): The sharpened image with enhanced edges.

## Notes
- Works in linear RGB in-place on the input tensor.
- The output is clamped to the [0,1] range.
- For a subtle effect, use small radius (2–4) and low strength (0.5–1.0).
- For stronger edge emphasis, increase radius and/or strength.

## Category
⭐StarNodes/Image And Latent
