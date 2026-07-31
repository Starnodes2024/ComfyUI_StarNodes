# ⭐ Star Video Joiner

## Overview

The **Star Video Joiner** node allows you to combine multiple image batches and audio inputs into a single unified output. This is particularly useful for preparing content for video creation nodes like VHS Video Combine.

## Features

- **5 Image Inputs**: Combine up to 5 image batches (input 1 is required, inputs 2-5 are optional)
- **5 Audio Inputs**: Combine up to 5 audio streams (all optional)
- **Automatic Image Resizing**: Images 2-5 are automatically resized to match the dimensions of image 1
- **Smart Audio Handling**: If the first audio input is longer than needed, it passes through unchanged
- **Batch Concatenation**: All image batches are concatenated along the batch dimension

## Inputs

### Required
- **image_1** (IMAGE): The primary image batch that sets the reference dimensions

### Optional
- **image_2** (IMAGE): Additional image batch (will be resized to match image_1)
- **image_3** (IMAGE): Additional image batch (will be resized to match image_1)
- **image_4** (IMAGE): Additional image batch (will be resized to match image_1)
- **image_5** (IMAGE): Additional image batch (will be resized to match image_1)
- **audio_1** (AUDIO): Primary audio input
- **audio_2** (AUDIO): Additional audio input
- **audio_3** (AUDIO): Additional audio input
- **audio_4** (AUDIO): Additional audio input
- **audio_5** (AUDIO): Additional audio input

## Outputs

- **images** (IMAGE): Combined image batch containing all input images
- **audio** (AUDIO): Combined audio stream (optional, only if audio inputs are provided)

## How It Works

### Image Processing
1. The first image input (`image_1`) is used as the reference for dimensions
2. All subsequent image inputs (2-5) are resized using bilinear interpolation to match the height and width of `image_1`
3. All images are concatenated along the batch dimension (dim=0)

### Audio Processing
1. All provided audio inputs are collected
2. If only one audio input is provided, it passes through unchanged
3. If multiple audio inputs are provided, they are concatenated along the time dimension
4. The sample rate from the first audio input is preserved

## Usage Example

### Basic Workflow
```
[Image Batch 1] ──┐
[Image Batch 2] ──┤
[Image Batch 3] ──┼──> [Star Video Joiner] ──> [VHS Video Combine]
[Audio 1] ────────┤                          ┌──>
[Audio 2] ────────┘                          └──>
```

### Common Use Cases

1. **Combining Multiple Video Segments**: Join several image sequences with their corresponding audio tracks
2. **Multi-source Video Creation**: Merge content from different generation nodes
3. **Audio Mixing**: Combine multiple audio tracks into a single stream
4. **Batch Processing**: Concatenate multiple image batches for longer video sequences

## Tips

- Always connect `image_1` as it's required and sets the reference dimensions
- If your images have different aspect ratios, they will be resized to match `image_1`
- Audio inputs are completely optional - you can use this node for images only
- The node automatically handles different audio formats (dict or tensor)

## Category

Located in: **⭐StarNodes/Video**

## Technical Details

- Uses PyTorch's `F.interpolate` with bilinear mode for image resizing
- Preserves image quality during resizing with `align_corners=False`
- Handles both dictionary-based and tensor-based audio formats
- Efficient batch concatenation using `torch.cat`
