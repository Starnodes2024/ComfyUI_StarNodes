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

## Outputs

| Output | Type | Notes |
|---|---|---|
| `video` | STAR_FILENAMES | file reference for the ⭐ Star Video Compressor |
| `images` | IMAGE | decoded frames (after all frame controls) |
| `audio` | AUDIO | soundtrack, cut to the kept frame range |
| `fps` | FLOAT | effective fps of the returned frames |
| `frames` | INT | number of returned frames |
| `info` | STRING | report (size, fps, frame counts, path) |
| `video_native` | VIDEO | native ComfyUI `VIDEO` (VideoInput), cut to the same range — connects to native nodes like SaveVideo |

## Widgets

- **video** — file from the input folder (upload button included).
- **force_rate** — resample fps, `0` = keep original.
- **skip_first_frames** — drop N frames from the start (applied after
  force_rate / every-kth).
- **select_every_kth** — keep every k-th frame (also divides the fps).
- **frame_load_cap** — max frames to load, `0` = all (RAM protection).
- **start_frame** — first frame to keep (0 = start).
- **end_frame** — end frame, exclusive (0 = to the end). The audio output is
  trimmed to the same time range whenever start/end/skip/cap cut the clip.

## Notes

- Frame ranges apply to the decoded sequence (after force_rate/every-kth)
  and clamp automatically if the video is shorter than expected.
- The Load button only probes metadata — nothing is decoded until the
  workflow runs.
