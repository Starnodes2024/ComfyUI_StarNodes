# ⭐ Star Video Joiner

## Overview

The **Star Video Joiner** node allows you to combine multiple image batches, videos and audio inputs into a single unified output. This is particularly useful for preparing content for video creation nodes like VHS Video Combine.

## Features

- **Dynamic Image Inputs**: Start with one image input — a new slot appears automatically each time you connect one (up to 20)
- **Dynamic Video Inputs**: Start with one video input — a new slot appears automatically each time you connect one (up to 20). Videos are decoded to image frames and joined together with the image inputs, so you can freely mix images and videos.
- **Dynamic Audio Inputs**: Start with one audio input — a new slot appears automatically each time you connect one (up to 20)
- **Automatic Image Resizing**: Everything after the first connected image/video is automatically resized to match the reference dimensions
- **Smart Audio Handling**: If only one audio input is provided, it passes through unchanged
- **Batch Concatenation**: All image/video frames are concatenated along the batch dimension

## Inputs

### Dynamic Image Slots (IMAGE)
- **image_1** (IMAGE): The first image batch — sets the reference dimensions if connected first
- **image_2** … **image_20** (IMAGE): Additional image batches, appear automatically when the previous slot is connected

### Dynamic Video Slots (STAR_FILENAMES)
- **video_1** (STAR_FILENAMES): First video input — accepts the `video` output from **Star Video Loader** (or **Star Video Compressor**), decoded to image frames via ffmpeg
- **video_2** … **video_20** (STAR_FILENAMES): Additional video inputs, appear automatically when the previous slot is connected. Frames from all connected videos are appended after the image inputs.

### Dynamic Audio Slots (AUDIO)
- **audio_1** (AUDIO): First audio input (optional)
- **audio_2** … **audio_20** (AUDIO): Additional audio inputs, appear automatically when the previous slot is connected

## Outputs

- **images** (IMAGE): Combined image batch containing all connected images and decoded video frames
- **audio** (AUDIO): Combined audio stream (only if audio inputs are provided)

## How It Works

### Image / Video Processing
1. Image inputs are processed first, then video inputs are decoded into frames by running ffmpeg on the file paths carried by the STAR_FILENAMES reference
2. The first item in that combined sequence sets the reference dimensions
3. Everything after it is resized using bilinear interpolation to match the reference height and width
4. All frames are concatenated along the batch dimension (dim=0)

### Audio Processing
1. All provided audio inputs are collected in connection order
2. If only one audio input is provided, it passes through unchanged
3. If multiple audio inputs are provided, they are concatenated along the time dimension
4. The sample rate from the first audio input is preserved

### Dynamic Input Growth
A ComfyUI frontend extension (`star_video_joiner_dynamic.js`) watches connections on the node. Each group (image/video/audio) tracks its own highest connected slot and always keeps exactly one empty slot after it, up to a maximum of 20 slots per group. The three groups are fully independent — connecting an audio input never affects the image or video slots, and vice versa.

## Usage Example

### Basic Workflow
```
[Image Batch 1] ──┐
[Image Batch 2] ──┤    (slot appears after connecting image_1)
[Star Video Loader: video] ─┤    (video slot appears after connecting video_1)
[Audio 1] ────────┼──> [Star Video Joiner] ──> [VHS Video Combine]
[Audio 2] ────────┘    (slot appears after connecting audio_1)
```

### Common Use Cases

1. **Combining Multiple Video Segments**: Join several image sequences with their corresponding audio tracks
2. **Multi-source Video Creation**: Merge content from different generation nodes
3. **Audio Mixing**: Combine multiple audio tracks into a single stream
4. **Batch Processing**: Concatenate multiple image batches for longer video sequences

## Tips

- The first connected image or video sets the reference dimensions — connect it first
- If your images/videos have different aspect ratios, they will be resized to match the reference
- Video and audio inputs are completely optional — you can use this node for images only
- The node automatically handles different audio formats (dict or tensor)
- New slots appear automatically — just connect and go, up to 20 inputs per group

## Category

Located in: **⭐StarNodes/Video**

## Technical Details

- Uses PyTorch's `F.interpolate` with bilinear mode for image resizing
- Preserves image quality during resizing with `align_corners=False`
- Handles both dictionary-based and tensor-based audio formats
- Efficient batch concatenation using `torch.cat`
- Video inputs use the shared `video_tools.star_nodes_common` ffmpeg helpers (`probe_media`, `run_ffmpeg_pipe`) to decode STAR_FILENAMES paths into raw RGB frames
- Dynamic inputs implemented via a frontend extension (`star_video_joiner_dynamic.js`) that adds/removes real input sockets, max 20 per group (image/video/audio), each group tracked independently by its own highest connected index
