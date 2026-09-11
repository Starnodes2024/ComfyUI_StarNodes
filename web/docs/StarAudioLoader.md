# ⭐ Star Audio Loader — Help

Loads an audio file (wav, flac, mp3, m4a, aac, ogg, opus, ...) from the
ComfyUI input folder (upload supported) and outputs the core AUDIO dict plus
the cut duration in seconds.

## Load, preview & cut (no workflow run needed)

1. Pick (or upload) an audio file.
2. Click **🔊 Load Audio** — the node probes the file in-place and fills an
   inline **`<audio>` preview** with playback controls. No workflow run
   needed.
3. Drag the **range slider** under the preview to cut the clip — the left
   handle sets the start time, the right handle the end time (in seconds,
   0.01 s precision). The track is color-coded (green = kept range between
   the handles, red = cut range), and the preview seeks to the handle you
   are dragging so you can hear exactly where you are cutting. Pressing
   **play** then plays only the cut range (it loops between start and end).
   The range stays in sync with the `start_time` / `end_time` number widgets
   (use those for exact entry).
4. Run the workflow — only the selected range is decoded.

Choosing another file resets the preview — click Load again.

## Outputs

| Output | Type | Notes |
|---|---|---|
| `audio` | AUDIO | core AUDIO dict, cut to the selected time range |
| `seconds` | INT | duration of the returned (cut) audio in whole seconds |
| `seconds_str` | STRING | duration of the returned audio as a decimal string (e.g. `12.345`) |
| `info` | STRING | report (codec, channels, sample rate, cut range, path) |

## Widgets

- **audio** — file from the input folder (upload button included).
- **start_time** — start of the cut in seconds (0 = start of the file).
- **end_time** — end of the cut in seconds (0 = to the end of the file).

## Notes

- The Load button only probes metadata — nothing is decoded until the
  workflow runs.
- Time ranges clamp automatically if the file is shorter than expected.
- The audio is decoded to 16-bit PCM via ffmpeg and returned as a float32
  waveform in the standard ComfyUI AUDIO format, so it drops straight into
  any audio workflow (SaveAudio, PreviewAudio, Star Sound Mixer, ...).
