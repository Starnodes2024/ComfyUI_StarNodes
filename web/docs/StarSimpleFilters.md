# Star Simple Filters

A simple but powerful node to apply common image adjustments.

## Features
- **Sharpen**: Increase image sharpness. Range: 0.0 to 100.0. Default: 0.0 (No effect).
- **Blur**: Apply Gaussian Blur. Range: 0.0 to 100.0. Default: 0.0 (No effect).
- **Saturation**: Adjust color intensity.
  - `0`: Neutral (Original)
  - `-100`: Black and White
  - `100`: Double saturation
- **Contrast**: Adjust contrast.
  - `0`: Neutral (Original)
  - Negative values decrease contrast.
  - Positive values increase contrast.
- **Brightness**: Adjust brightness.
  - `0`: Neutral (Original)
  - Negative values darken the image.
  - Positive values lighten the image.
- **Temperature**: Adjust color temperature.
  - `0`: Neutral
  - Negative values (`-100`): Cooler (Blue shift)
  - Positive values (`100`): Warmer (Red shift)
- **Filter Strength**: Global opacity of the effect.
  - `100`: Full effect.
  - `0`: Original image.

## Usage
Connect an image to the `image` input. Adjust sliders to taste. The node processes images in batches if provided.

## Category
`⭐StarNodes/Image`
