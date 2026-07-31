# ⭐ Star Radial Blur

## Description
Applies a radial blur similar to Photoshop's Radial Blur filter. Supports **Zoom** and **Spin** modes:
- **Zoom**: samples along rays that converge toward a center point, creating a dynamic zoom/warp effect.
- **Spin**: rotates samples around a center point, creating circular motion blur.

## Inputs
- **image** (IMAGE, required): Source image.
- **mode** (CHOICE, required): Radial blur mode. `Zoom` (zoom/stretch effect) or `Spin` (circular motion blur). Default: `Zoom`.
- **strength** (FLOAT, required): Overall blur strength. Higher values increase zoom/stretching (Zoom) or rotation amount (Spin). Default: 0.5.
- **samples** (INT, required): Number of samples taken along each ray. More samples = smoother blur but slower. Default: 16.
- **center_x** (FLOAT, required): Horizontal blur center, normalized 0.0 (left) to 1.0 (right). Default: 0.5.
- **center_y** (FLOAT, required): Vertical blur center, normalized 0.0 (top) to 1.0 (bottom). Default: 0.5.

## Outputs
- **image** (IMAGE): Image with radial zoom blur applied.

## Notes
- For **Zoom**: set **strength** low (0.2–0.6) and **samples** around 8–24 for a subtle motion/zoom effect.
- For **Spin**: start with **strength** around 0.2–0.5 and **samples** around 8–24 for smooth circular motion blur.
- Moving **center_x / center_y** off-center creates directional/spiral effects in both modes.
- This runs fully on GPU using PyTorch and works with batched images.
