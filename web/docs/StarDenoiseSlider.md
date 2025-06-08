# Star Denoise Slider

## Description
A simple slider node for controlling denoise strength values in sampling workflows. This node provides an intuitive interface for adjusting denoising strength with standardized steps and outputs both float and string formats for compatibility with different nodes.

## Inputs
- **denoise_value**: Slider to select denoise strength (range: 0.1 to 1.0, default: 0.7, step: 0.05)

## Outputs
- **denoise**: The selected denoise value as a float
- **denoise_str**: The selected denoise value as a string

## Usage
1. Add the node to your workflow
2. Adjust the slider to set your desired denoising strength
3. Connect the appropriate output (float or string) to compatible nodes that require a denoise value

This node is useful for:
- Creating consistent denoising strength controls across workflows
- Providing a visual slider interface for fine-tuning denoise values
- Outputting both numeric and string formats for maximum compatibility
- Standardizing denoise values to 0.05 steps for predictable results

The node ensures values are properly rounded to the nearest 0.05 step and constrained within the valid range of 0.1 to 1.0.
