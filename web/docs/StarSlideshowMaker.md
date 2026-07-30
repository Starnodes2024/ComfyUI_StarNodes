# Star Slideshow Maker

Create and encode an image slideshow directly in one compact node. Frames are streamed one by one to FFmpeg, so long videos do not allocate a large ComfyUI `IMAGE` batch in RAM.

## Quick start

1. Connect one or more `IMAGE` outputs, or set `image_folder`.
2. Choose the aspect ratio, resolution, duration, timing, transition, and motion settings.
3. Set the encoder quality or target file size.
4. Run the node. The encoded video appears in the inline preview and is returned as `STAR_FILENAMES`.

## Image sources

### Dynamic connectors

Connect an image to `image_1`. When the last image connector is used, the frontend adds `image_2`, `image_3`, and so on. Image batches are flattened in connector order.

### Image folder

`image_folder` can be an absolute path or a path relative to the ComfyUI input folder. Files ending in `.jpg`, `.jpeg`, `.png`, or `.webp` are loaded in natural filename order; subfolders are not searched.

Connected images are used first, followed by folder images. If only a folder is used, its first file in natural order becomes the first slideshow image.

## Size and fit

- `aspect_ratio = auto` chooses the listed ratio closest to the first image.
- Manual ratios: `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, and `21:9`.
- `HD` uses a 1280-pixel long edge.
- `Full HD` uses a 1920-pixel long edge.
- `contain` fits the complete image inside the frame.
- `cover` fills the frame and center-crops overflow.
- `background` selects the letterbox color for `contain`.

## Duration and timing

- `duration_mode = fixed` uses `fixed_duration`.
- `duration_mode = audio` matches the connected audio duration.
- `seconds_per_image` repeats the image sequence until the total duration is filled.
- `split_total_duration` divides the total duration equally among all images.

## Transitions and motion

Transitions are placed at the end of each image segment and are clamped so they cannot consume almost the whole segment.

Available transitions:

- `none`
- `fade`
- `morph`
- `slide_left`, `slide_right`, `slide_up`, `slide_down`
- `wipe_left`, `wipe_right`, `wipe_up`, `wipe_down`
- `zoom`
- `pixelate`
- `random`

Available motion effects:

- `none`
- `zoom_in`, `zoom_out`
- `pan_left`, `pan_right`, `pan_up`, `pan_down`
- `random`

Morph uses OpenCV optical flow when available and otherwise falls back to a built-in blur/warp morph-style transition.

### Random transition / motion

Setting `transition` and/or `motion_effect` to `random` picks a different effect (excluding `random` itself) for every image instead of using the same one throughout. `seed` controls the pick: `0` re-rolls new random effects on every run, while any other value reproduces the same sequence of random effects across runs.

## Encoding

The node includes the compressor-style encoding settings:

- `quality`: compression quality from 0 to 100. Higher is better quality and a larger file.
- `format`: H.264 MP4, H.265 MP4, VP9 WebM, or AV1 MP4, depending on the encoders available in your FFmpeg build.
- `preset`: encoder speed versus efficiency. Slower usually creates a smaller file at the same quality.
- `filename_prefix`: output name, optionally with subfolders.
- `target_size_mb`: desired output size in MiB. Set to `0` to use the quality slider. When greater than `0`, target size wins and two-pass encoding is used where supported.
- `save_audio`: muxes connected audio into the output.
- `save_output`: saves to the ComfyUI output folder when enabled, or the temp folder when disabled.

The `Filenames` output uses the `STAR_FILENAMES` type and can be connected to other Star nodes that accept videos.

## RAM behavior

The node renders each frame and immediately sends it to FFmpeg. It does not create a complete float32 frame batch. This avoids the very large allocations that can occur when a long Full-HD slideshow is returned as a ComfyUI `IMAGE` batch.

## Requirements

Install the packages in `requirements.txt` into ComfyUI's Python environment. FFmpeg and FFprobe must be available on the system path. OpenCV is optional and only improves the morph transition.
