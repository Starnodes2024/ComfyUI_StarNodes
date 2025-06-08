# Detail Star Daemon

## Description
The Detail Star Daemon is a specialized node that creates detail enhancement schedules for use with StarSampler nodes. Based on the Detail Daemon technology, it allows for precise control over how and when detail enhancement is applied during the diffusion sampling process. This node doesn't perform sampling itself but creates a configuration that StarSampler nodes can use to enhance fine details in generated images.

## Inputs

### Required
- **detail_amount**: The strength of the detail enhancement effect (-5.0 to 5.0, default: 0.1)
- **detail_start**: When in the sampling process to start applying enhancement (0.0 to 1.0, default: 0.2)
- **detail_end**: When in the sampling process to end enhancement (0.0 to 1.0, default: 0.8)
- **detail_bias**: Controls the midpoint of the enhancement curve (0.0 to 1.0, default: 0.5)
- **detail_exponent**: Controls the shape of the enhancement curve (0.0 to 10.0, default: 1.0)

## Outputs
- **detail_schedule**: A detail schedule configuration that can be connected to compatible StarSampler nodes

## Usage
1. Add the Detail Star Daemon node to your workflow
2. Configure the detail enhancement parameters
3. Connect the detail_schedule output to a compatible sampler node's detail_schedule input
4. The sampler will use this schedule to enhance details during the sampling process

## Features

### Fine-Grained Control
- **Strength Control**: Adjust the intensity of detail enhancement with detail_amount
- **Timing Control**: Set precisely when enhancement begins and ends in the sampling process
- **Curve Shaping**: Customize the enhancement curve with bias and exponent parameters

### Compatible with Multiple Samplers
Works with the following StarSampler nodes:
- StarSampler SD/SDXL/SD3.5
- StarSampler FLUX

## Technical Details
- The detail schedule creates a cosine-based curve that modifies the noise schedule during diffusion
- Positive detail_amount values enhance details, while negative values can smooth or reduce details
- The enhancement is strongest at the midpoint between detail_start and detail_end, determined by detail_bias
- The detail_exponent parameter controls how sharply the enhancement effect ramps up and down

## Parameter Guide

### detail_amount
- **Positive values (0.1 to 5.0)**: Enhance details, higher values create stronger enhancement
- **Negative values (-5.0 to 0)**: Reduce details, useful for smoothing or removing noise
- **Recommended range**: 0.05 to 0.3 for subtle enhancement, 0.3 to 1.0 for stronger effects

### detail_start and detail_end
- These define the portion of the sampling process where enhancement is applied
- Values are normalized from 0.0 (beginning of sampling) to 1.0 (end of sampling)
- **Recommended settings**:
  - For general detail enhancement: start=0.2, end=0.8
  - For early detail formation: start=0.1, end=0.5
  - For late detail refinement: start=0.5, end=0.9

### detail_bias
- Controls where the peak enhancement occurs between start and end
- 0.5 means the peak is exactly in the middle
- Lower values shift the peak earlier, higher values shift it later

### detail_exponent
- Controls the shape of the enhancement curve
- 1.0 creates a smooth cosine curve
- Higher values create a steeper, more focused peak
- Lower values create a broader, more gradual effect

## Notes
- For best results, use with 20+ sampling steps
- Detail enhancement works best with the "euler" sampler and "normal" or "karras" scheduler
- Subtle enhancements (0.05-0.2) often produce more natural results than stronger values
- The effect is multiplied by the CFG scale in the sampler, so higher CFG values will amplify the enhancement
- Experiment with different start/end ranges to target specific types of details
