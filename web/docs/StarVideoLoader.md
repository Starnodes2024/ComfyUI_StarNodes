# ⭐ Star Video Loader — Help

Loads a video from the ComfyUI input folder (upload supported) and outputs
the file reference, decoded frames, audio, fps and frame count.

## Load, preview & cut (no workflow run needed)

1. Pick (or upload) a video.
2. Click **📼 Load Video** — the node probes the file in-place and fills the
   **450×450 preview stage** (placeholder until then; the video letterboxes
   inside it, portrait or landscape — resize the node taller and the stage
   grows with it). No workflow run needed and no second preview — the
   frontend's built-in upload preview is hidden.
3. Drag the **range slider** under the preview to cut the clip — the left
   handle sets the start frame, the right handle the end frame. The track is
   color-coded (green = kept frames between the handles, red = cut frames),
   and the preview seeks to the handle you are dragging so you can see
   exactly which frame you are cutting at. Pressing **play** then plays only
   the cut range (it loops between start and end). The range stays in sync
   with the `start_frame` / `end_frame` number widgets (use those for exact
   entry).
4. Run the workflow — only the selected range is decoded, and the **audio is
   cut to the same range** so picture and sound stay aligned.

Choosing another video resets the preview — click Load again.

## Crop (no workflow run needed)

1. Click **📼 Load Video** first (the crop dialog needs the probed file).
2. Click **✂ Crop** — a dialog opens showing the video in its own aspect
   ratio, scaled to fit the screen (max 1024 px per side).
3. Drag the **slider** under the preview to scrub to a frame where the
   subject is clearly visible.
4. **Drag a box** over the frame — that's the crop area. The label in the
   box's corner shows the pixel size the output will have. Drag again to
   redraw; a smaller-than-1% box is ignored.
5. **Apply** stores the region — Cancel closes without changes and
   **Delete** removes a previously stored crop. A small `crop WxH%` badge
   appears on the node — click its **✕** to remove the crop, or clear the
   `crop_rect` widget text manually.

The node's inline preview also updates: it shows **only the cropped area**
(the box takes the crop's shape and the video is zoomed to fill it), so you
can scrub through and check the framing before running. Removing the crop
restores the normal letterboxed preview.

When the workflow runs, ffmpeg crops inside the decode pipeline, so the
`images` output is exactly the drawn size — cropping also makes decoding
faster and uses less RAM. The crop is stored as fractions of the frame, so
it survives resolution differences between preview and source. Switching
to another video clears the crop. The inline preview is aspect-fitted to the
video (or the crop) and never exceeds 1024 px in either dimension. Note:
`video_native` passes the file
through for native nodes and stays uncropped.

## Outputs

| Output | Type | Notes |
|---|---|---|
| `video` | STAR_FILENAMES | file reference for the ⭐ Star Video Compressor |
| `images` | IMAGE | decoded frames (after all frame controls + crop) |
| `audio` | AUDIO | soundtrack, cut to the kept frame range |
| `fps` | FLOAT | effective fps of the returned frames (the `force_rate` value when set) |
| `frames` | INT | number of returned frames |
| `info` | STRING | report (size, fps, frame counts, path) |
| `video_native` | VIDEO | native ComfyUI `VIDEO` (VideoInput), cut to the same range — connects to native nodes like SaveVideo |
| `seconds` | FLOAT | duration of the returned frames in seconds (`frames ÷ fps`) — reflects all trims, frame controls and the crop |

## Widgets

- **video** — file from the input folder (upload button included).
- **force_rate** — `original` / `24` / `25` / `30`. Converts the video to
  the chosen frame rate by **duplicating or dropping frames spread
  evenly across the clip** (no blending), so motion stays as fluid as
  possible. E.g. 24 → 30 inserts 6 duplicated frames per second, 30 → 24
  removes 6. `original` keeps the video's own rate. The `fps` and
  `frames` outputs reflect the new rate and frame count.
- **skip_first_frames** — drop N frames from the start (applied after
  force_rate / every-kth).
- **select_every_kth** — keep every k-th frame (also divides the fps).
- **frame_load_cap** — max frames to load, `0` = all (RAM protection).
- **start_frame** — first frame to keep (0 = start).
- **end_frame** — end frame, exclusive (0 = to the end). The audio output is
  trimmed to the same time range whenever start/end/skip/cap cut the clip.
- **crop_rect** — crop region as `x,y,w,h` fractions of the frame (0–1),
  set by the **✂ Crop** button. Empty = full frame; editable by hand.
- **resize** — `original` / `0.5`–`3.0 MP` / `HD (1280x720)` /
  `Full HD (1920x1080)`. Resizes the decoded frames **after the crop**,
  inside ffmpeg (no extra decode cost): MP presets scale the frame to that
  pixel area, HD/Full HD fit it inside the box — aspect preserved, sizes
  rounded to even. `original` keeps the source/crop size. Only affects the
  `images` output.
- **minimax_snap** — when on, the output length is snapped down to the
  MiniMax H3 `17n+5` frame grid (5, 22, 39, 56, … frames): the trim sliders
  and start/end fields only produce lengths a Minimax node renders without
  padding, and `seconds`/`frames` reflect the snapped count. A non-aligned
  tail is unusable for Minimax anyway, so nothing is lost. Off = free
  cutting.

## Notes

- Frame ranges apply to the decoded sequence (after force_rate/every-kth)
  and clamp automatically if the video is shorter than expected.
- The Load button only probes metadata — nothing is decoded until the
  workflow runs.
