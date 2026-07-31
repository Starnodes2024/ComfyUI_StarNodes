# ⭐ Star Image Loop

Creates seamless looping video frames from panoramic images. Perfect for creating social media content from AI-generated or photographed panoramas. Connect the output to any Video Combine node to save the final video.

**Supports multiple image inputs!** Connect additional images to join them horizontally into a longer panorama.

- __Category__: `⭐StarNodes/Video`
- __Class__: `StarImageLoop`
- __File__: `star_image_loop.py`

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
- __image 1__ (IMAGE): First panorama image. Always visible.
- __image 2, image 3, ...__ (IMAGE): Additional images that appear dynamically when you connect more inputs. Images are joined horizontally in order (image 1 → image 2 → image 3 → ...).

## Outputs
- __images__ (IMAGE): Batch of video frames `[N, H, W, C]` ready for a Video Combine node.
- __fps__ (INT): The fps value, passed through for use with Video Combine nodes.

## Behavior
- **Multiple images** are joined horizontally in order, scaled to match the output height.
- The combined panorama is tiled horizontally to create a seamless loop.
- Frames are generated at the specified fps and duration (e.g., 24fps × 10s = 240 frames).
- All dimensions are ensured to be even numbers for video encoding compatibility.

## Usage Tips
- For best results, use panoramic images with a 2:1 aspect ratio (e.g., 4096x2048).
- **Combine multiple images** to create longer panoramas - great for stitching AI-generated scenes together!
- The seamless loop works because the panorama scrolls exactly one combined image width per video duration.
- Use 9:16 ratio for TikTok/Reels, 1:1 for Instagram feed, 16:9 for YouTube.
- Higher FPS (30-60) creates smoother motion but more frames.
- Longer durations create slower, more relaxing pans.

## Example Workflow
1. Load a panoramic image using any image loader node
2. Connect to **Star Image Loop**
3. Set resolution to "Full HD (1920)" and ratio to "9:16 (TikTok/Reels)"
4. Set duration to 15 seconds for a relaxing pan
5. Connect the `images` output to a **Video Combine** node (e.g., from VHS or other video nodes)
6. Connect the `fps` output to the Video Combine node's frame_rate input
7. Run the workflow

## Version
- Introduced in StarNodes v1.9.4
