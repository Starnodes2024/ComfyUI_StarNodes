# Star Video Compare

## Description

**Star Video Compare** is an interactive video comparison node. Connect two video sources and use a draggable wipe slider to compare them directly inside the node. It also produces a stitched comparison video (side-by-side or top/bottom) as an IMAGE batch that can be piped to any save or compress node.

## Inputs

### Required Widgets
- **layout**: Orientation of the stitched comparison video — `left/right` (side by side) or `top/bottom` (stacked).
- **max_size**: Longest side of the stitched output in pixels. The output is downscaled if it exceeds this. `0` = no resize.
- **fps**: Frame rate for the preview playback and stitched output.
- **loop**: Whether the preview video loops.
- **caption_video1**: Caption text for video 1 (drawn on the stitched output).
- **caption_video2**: Caption text for video 2 (drawn on the stitched output).
- **compare_position**: Initial slider position from `0.0` (Video 2) to `1.0` (Video 1). Hidden — updated when you drag the slider.

### Optional Connectors
- **video_1**: First video as an IMAGE batch (frames). Takes priority over `video_native_1`.
- **video_2**: Second video as an IMAGE batch (frames). Takes priority over `video_native_2`.
- **video_native_1**: Native ComfyUI VIDEO input for video 1. Used when `video_1` is not connected.
- **video_native_2**: Native ComfyUI VIDEO input for video 2. Used when `video_2` is not connected.

## Outputs

- **comparison_video**: IMAGE batch of the stitched comparison video. Both videos are placed side-by-side (left/right) or stacked (top/bottom) with optional caption bars below each segment. Connect to a video save or compress node to export.

## How to Use

1. Connect two video sources — either IMAGE batches or native VIDEO inputs (or mix).
2. Set the layout (side-by-side or top/bottom) and max output size.
3. Optionally add captions for each video.
4. Execute the workflow.
5. Drag the vertical divider or the slider at the bottom of the node to compare.
6. Both videos play in sync automatically.

## Notes

- If the two videos have different resolutions, the smaller one is **upscaled** (Lanczos) to match the larger.
- If the two videos have different frame counts, the shorter one is **looped** to match the longer one in the stitched output.
- Preview videos are saved as temporary H.264 MP4s for the node's interactive viewer.
- The slider supports zoom (mouse wheel) and pan (right/middle click or Shift+click), with double-click to reset the view.
- Captions are drawn as white text on a black bar below each video segment in the stitched output.
