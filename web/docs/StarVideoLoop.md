# ⭐ Star Video Loop

Creates seamless looping video frames from video inputs. Videos are scrolled horizontally to create a sliding panorama effect with moving content. Connect the output to any Video Combine node to save the final video.

**Supports multiple video inputs!** Connect additional videos to join them horizontally into a longer sliding video panorama.

- __Category__: `⭐StarNodes/Video`
- __Class__: `StarVideoLoop`
- __File__: `star_video_loop.py`

## Inputs

### Required
- __resolution__ (CHOICE, required, default: "Full HD (1920)"): Output frame resolution width.
  - `HD (1280)` - 1280px width
  - `Full HD (1920)` - 1920px width
  - `2K (2560)` - 2560px width
  - `4K (3840)` - 3840px width
- __aspect_ratio__ (CHOICE, required, default: "1:1 (Square)"): Output frame aspect ratio.
  - `1:1 (Square)` - Square format
  - `9:16 (TikTok/Reels)` - Vertical video for TikTok, Instagram Reels, YouTube Shorts
  - `4:5 (Instagram)` - Instagram portrait format
  - `16:9 (YouTube)` - Standard widescreen
  - `3:4 (Portrait)` - Classic portrait
  - `2:3 (Portrait)` - Tall portrait
- __fps__ (INT, required, default: 24): Frames per second (1-60). Passed through to output for Video Combine nodes.
- __duration__ (FLOAT, required, default: 10.0): Video duration in seconds (1-300).
- __direction__ (CHOICE, required, default: "Left to Right"): Scroll direction.
  - `Left to Right` - Pan from left to right
  - `Right to Left` - Pan from right to left

### Optional (Dynamic)
- __video 1__ (IMAGE): First video as IMAGE batch. Always visible.
- __video 2, video 3, ...__ (IMAGE): Additional videos that appear dynamically when you connect more inputs. Videos are joined horizontally in order (video 1 → video 2 → video 3 → ...).

## Outputs
- __images__ (IMAGE): Batch of video frames `[N, H, W, C]` ready for a Video Combine node.
- __fps__ (INT): The fps value, passed through for use with Video Combine nodes.

## Behavior
- **Multiple videos** are joined horizontally in order, scaled to match the output height.
- Videos play simultaneously while scrolling horizontally.
- Shorter videos will loop to match the duration.
- The combined video panorama scrolls seamlessly in a loop.
- Frames are generated at the specified fps and duration.

## Usage Tips
- For best results, use videos that loop seamlessly themselves.
- **Combine multiple videos** to create longer panoramas with moving content!
- Use 9:16 ratio for TikTok/Reels, 1:1 for Instagram feed, 16:9 for YouTube.
- Higher FPS (30-60) creates smoother motion but more frames.
- Longer durations create slower, more relaxing pans.

## Example Workflow
1. Load videos using a Load Video node (outputs IMAGE batch)
2. Connect to **Star Video Loop**
3. Set resolution to "Full HD (1920)" and ratio to "9:16 (TikTok/Reels)"
4. Set duration to 15 seconds for a relaxing pan
5. Connect the `images` output to a **Video Combine** node
6. Connect the `fps` output to the Video Combine node's frame_rate input
7. Run the workflow

## Version
- Introduced in StarNodes v1.9.4
