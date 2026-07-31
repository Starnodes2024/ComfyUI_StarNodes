# ⭐ Star Black & White

## Description
Photoshop-style black and white conversion node. Lets you mix the Red, Green, and Blue channels with individual weights (like the B&W adjustment in Photoshop) and optionally add film-style grain.

## Inputs
- **image** (IMAGE, required): Source image.
- **preset** (CHOICE, required): Named B&W look. Selecting a preset updates the channel-mix and grain sliders to a good starting point; you can then tweak them freely. Default: `Neutral Luminance`.
- **red_weight** (FLOAT, required): Contribution of the Red channel to the luminance mix. Default: 0.30.
- **green_weight** (FLOAT, required): Contribution of the Green channel to the luminance mix. Default: 0.59.
- **blue_weight** (FLOAT, required): Contribution of the Blue channel to the luminance mix. Default: 0.11.
- **normalize_weights** (BOOL, required): When enabled, the three weights are normalized to sum to 1.0 so exposure stays consistent when you move sliders. Default: True.
- **add_grain** (BOOL, required): Enable/disable film grain overlay. Default: False.
- **grain_strength** (FLOAT, required): Strength of the grain added on top of the grayscale image. Higher values = more visible grain. Default: 0.25.
- **grain_density** (FLOAT, required): Overall noise density/contrast. Lower values push grain closer to mid-gray (subtle). Higher values keep full contrast. Default: 0.8.
- **grain_size** (INT, required): Controls grain coarseness. Higher values = larger grain (lower-res noise). Default: 4.

## Outputs
- **image** (IMAGE): 3-channel grayscale image with optional grain.

## Notes
- The default weights (0.30 / 0.59 / 0.11) roughly match a standard luminance conversion.
- To emulate channel-based B&W looks, push one channel up and others down (e.g. high Red + low Blue for skin-friendly conversions).
- Grain is generated procedurally on the GPU using low-res noise upsampled to image size, so it works with batches and large images.

### Presets
- **Neutral Luminance**: Classic luminance mix (0.30 / 0.59 / 0.11), no grain.
- **Skin High Red**: Higher red contribution, lower blue; good for smoother skin tones.
- **Sky High Blue**: Emphasizes blue channel for dramatic skies and clouds.
- **Green Filter**: Strong green weighting, similar to a green color filter in B&W photography.
- **High Contrast**: Balanced red/green with less blue, for punchier contrast.
- **Soft Matte**: Slightly shifted mix for a softer, flatter B&W base.
- **Film Grain Subtle**: Neutral mix with subtle fine grain.
- **Film Grain Strong**: Slightly warmer mix with stronger, more visible grain.
