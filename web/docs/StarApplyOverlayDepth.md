# Star Apply Overlay (Depth)

## Description
Blends a filtered image over a source image using a mask. The mask can be provided directly or derived from a depth/greyscale image. Includes optional inversion, preview saving, and pixel-based Gaussian blur of the mask.

## Category
⭐StarNodes/Image And Latent

## Inputs
- source_image (IMAGE): Base image
- filtered_image (IMAGE): Image to overlay on top
- strength (INT 0–100, default 100): Blend amount scaled by the mask

Optional:
- depth_image (IMAGE): Greyscale/RGB image used to derive the mask (ignored if `mask` is provided)
- mask (MASK or IMAGE): Mask in 0..1; if IMAGE, it will be converted to luminance
- invert_mask (BOOLEAN, default False): Invert 0..1 mask
- show_preview (BOOLEAN, default True): Save a small preview of the mask in outputs/StarOverlay/MaskPreviews
- blur_mask_px (INT, default 0): Pixel radius for Gaussian blur (approx sigma ≈ radius/3)

## Outputs
- image (IMAGE): Composited image result

## Behavior & Details
- If both `mask` and `depth_image` are provided, `mask` takes precedence
- All inputs are resized to match `source_image` resolution
- Batch sizes are auto-matched; single-batch inputs are repeated to the largest batch size
- Luminance conversion for masks: 0.299R + 0.587G + 0.114B
- Blending formula: `out = source * (1 - alpha) + filtered * alpha`, where `alpha = (strength/100) * mask`

## Usage
1. Provide `source_image` and `filtered_image`
2. Connect either a `mask` or a `depth_image`
3. Adjust `strength`, optionally toggle `invert_mask`
4. Set `blur_mask_px` to soften edges; enable `show_preview` for a quick UI preview image

## Notes
- Preview images are saved under `outputs/StarOverlay/MaskPreviews`
- Ensure mask is in 0..1 range; non-floating types are handled and converted internally
