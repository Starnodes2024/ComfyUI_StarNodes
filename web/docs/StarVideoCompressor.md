# Star Video Compressor — Help (v2.3, standalone)

A ComfyUI custom node pack that loads and compresses videos — built for
cases like Discord's 10 MB upload limit.

**Fully standalone**: does NOT require Video Helper Suite. Ships its own
video type (`STAR_FILENAMES`), a full **Star Video Loader** node (file +
frames + audio + fps + frame count), a direct path widget, and uses
ComfyUI **core AUDIO**. Progress is shown three ways: the standard ComfyUI
UI progress bar (like KSampler), a **fancy animated DOM progress bar** on
the node itself, and a **% bar in the console**.

Nodes (category **video**):

| Node | Purpose |
|---|---|
| **Star Video Loader** | Load a video from the `input` folder → file ref, IMAGE frames, AUDIO, fps, frame count, info. |
| **Star Video Compressor** | Compress video(s) or an `IMAGE` batch → `STAR_FILENAMES` + info + inline preview. |

---

## 1. Quick start

**A. Compress a video file end-to-end (only these two nodes):**

```
[Star Video Loader] ──(video)──► [Star Video Compressor]
                                      target_size_mb: 9.5   ◄── for Discord
                                      format: video/h264-mp4
                                      filename_prefix: discord/my_clip
```

**B. Round-trip through an image workflow (upscale, filter, …):**

```
[Star Video Loader] ──(images)──► [your image nodes] ──► [Star Video Compressor]
                  └─(fps)──────► (frame_rate input) ────┘   (images input)
                  └─(audio)──────────────────────────────►  (audio input)
```

Tip for B: right-click the compressor → **Convert frame_rate to input**,
then connect the loader's **fps** output so the re-encode keeps the exact
source frame rate.

**C. No loader at all:** type a path into the compressor's `video_path`
widget (absolute, or relative to the ComfyUI `input` folder).

---

## 2. Star Video Loader

### Widgets

| Widget | Default | Description |
|---|---|---|
| `video` | — | Video file from the ComfyUI `input` folder (subfolders included; upload supported on current frontends). Extensions: mp4, webm, mkv, mov, avi, m4v, mpg, mpeg, gif, wmv, flv, ts. |
| `force_rate` | 0.0 | Resample to this fps. `0` = keep original frame rate. |
| `skip_first_frames` | 0 | Drop this many frames from the start (after resampling). |
| `select_every_kth` | 1 | Keep only every k-th frame; also divides the output fps. |
| `frame_load_cap` | 0 | Max frames to load (`0` = all). Protects RAM with long videos. |

### Outputs

| Output | Type | Description |
|---|---|---|
| `video` | `STAR_FILENAMES` | File reference — connect to the compressor's `video` input. |
| `images` | `IMAGE` | Decoded frames as a standard image batch (float 0–1, N×H×W×3). |
| `audio` | `AUDIO` | The video's audio track as ComfyUI core audio (empty if the video has none). |
| `fps` | `FLOAT` | **Effective** fps of the `images` output (after force_rate / every-kth). |
| `frames` | `INT` | Number of frames actually returned in `images`. |
| `info` | `STRING` | Report: resolution, fps, duration, codecs, size, frames loaded/decoded/total, audio sample rate, path. |

Note: decoding happens through ffmpeg directly into RAM — use
`frame_load_cap` / `select_every_kth` for long or high-resolution videos.

---

## 3. Star Video Compressor

### Inputs

| Input | Type | Description |
|---|---|---|
| `video` | `STAR_FILENAMES` | Video file(s) from the **Star Video Loader** (several files = batch compress). |
| `images` | `IMAGE` | Alternative: image batch, encoded at `frame_rate`. |
| `audio` | `AUDIO` | External audio track (core audio — e.g. built-in **LoadAudio** node or the loader's `audio` output). |

**Input priority:** `video` → `video_path` widget → `images`.

### Widgets

| Widget | Default | Description |
|---|---|---|
| `quality` | 60 | Slider 0–100. Higher = better/bigger. **Ignored when `target_size_mb` > 0.** |
| `format` | `video/h264-mp4` | Container + codec (see §5). |
| `preset` | medium | Speed vs efficiency. `slow`/`slower` = smaller files, longer encode. |
| `filename_prefix` | `StarVideo` | Output name without extension. Subfolders allowed (`discord/my_clip`), auto-created, never overwrites. |
| `target_size_mb` | 0.0 | Desired size in **MiB**. `0` = off; `> 0` **overrides** quality (two-pass bitrate mode). |
| `video_path` | *(empty)* | Direct path to a video file (absolute or relative to `input`). |
| `frame_rate` | 30.0 | Only for the `images` input. Convert to input and connect loader `fps` for exact timing. |
| `save_audio` | true | Keep/re-encode audio (AAC 128k; Opus for WebM). `false` = strip (saves ~16 KB/s). |
| `save_output` | true | `true` = `output` folder, `false` = `temp`. |

### Outputs

| Output | Type | Description |
|---|---|---|
| `Filenames` | `STAR_FILENAMES` | Compressed file(s), chainable. |
| `info` | `STRING` | Detailed per-file report (example in §6), also printed to console. |

---

## 4. Quality vs. target size — which one wins?

The modes are **mutually exclusive**, with a clear priority:

| `target_size_mb` | Active mode | What drives the encode |
|---|---|---|
| `0` | **Quality mode** | The `quality` slider → constant-quality CRF encode. |
| `> 0` | **Target size mode** | The target: bitrate = `(target×8×1024²×0.95 − audio) ÷ duration`, encoded in **two passes** (H.264/H.265/VP9). The quality slider is **ignored** — the info output says so. |

They cannot be combined — "exactly this quality" and "exactly this size"
are contradictory goals, so the target takes precedence.

- Quality starting points: **80–90** near-lossless · **60** default ·
  **30–45** small with visible loss.
- **Discord:** `target_size_mb: 9.5` (or `9.0` for extra margin under the
  10 MB limit). Results usually land within ±10 %; simple/short content
  tends to undershoot, complex motion can slightly overshoot.
- Very long video + tiny target hits the 32 kbps bitrate floor — trim or
  reduce resolution/fps upstream (the loader's `select_every_kth` /
  `force_rate` help here).

---

## 5. Formats

| Format | Ext | Codec | Notes |
|---|---|---|---|
| `video/h264-mp4` | .mp4 | H.264 | **Best compatibility** — Discord embeds, phones, browsers. Default. |
| `video/h265-mp4` | .mp4 | H.265/HEVC | ~25–50 % smaller at equal quality, slower. `hvc1` tag for Apple. |
| `video/vp9-webm` | .webm | VP9 | Open format, browser/Discord friendly, audio = Opus. |
| `video/av1-mp4` | .mp4 | AV1 (SVT-AV1) | Best compression, slow; needs recent ffmpeg. Target mode = single-pass VBR. |

---

## 6. Progress & info

While a node runs you get **three** progress indicators:

1. **DOM progress bar** on the node itself — animated gradient bar with
   live percentage and a sub-line (`pass 1/2 | 1.9/3.0s | 152 fps | 2.4x`),
   turning green when done.
2. **ComfyUI UI progress bar** (the standard one, like KSampler).
3. **Console % bar**:
   `[StarNodes] compressing [================>-----------]  62.4% | 1.9s/3.0s | 152 fps`

Example **info** output (target mode):

```
settings: format=video/h264-mp4 | preset=fast | target size 9.5 MiB (quality slider ignored) | audio=on

[1] my_clip.mp4
    in : render.mp4 | 1920x1080 @ 30.00 fps | 12.03 s | h264 | 84.20 MiB
    out: 1920x1080 @ 30.00 fps | 12.03 s | h264+aac | 9.41 MiB | overall 6470 kb/s
    mode: target size | 2-pass | preset fast | result 9.41 MiB (-0.9% vs target)
    size: 84.20 -> 9.41 MiB (-88.8%, 89% smaller)
    time: 21.3 s | saved to: /.../output/discord/my_clip.mp4
```

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Failed to convert an input value to a FLOAT value: frame_rate` | Bug in v2.0 (frame_rate was an optional widget). **Update to v2.1+** — or just re-enter the value. |
| `different bitdepth setting than first pass (8 vs 10)` | The source is a **10-bit** video (common for AI-generated HEVC clips) and pass 1 ran in 10-bit while pass 2 forced 8-bit. **Fixed in v2.2** — both passes now use the same pixel format. |
| Duplicate/overflowing video preview on the loader | The frontend already previews the selected video; the extra custom preview was removed in **v2.2**. The compressor now reserves a fixed placeholder box and snaps to the video's aspect ratio, so nothing spills outside the node. |
| `ffmpeg was not found` | `pip install imageio-ffmpeg` (see requirements.txt) or install system ffmpeg. |
| `your ffmpeg build does not include the encoder …` | Pick `video/h264-mp4` or install a fuller ffmpeg build. |
| `could not determine the input duration` | Exotic/corrupt input — re-encode first, or use quality mode. |
| Loader eats too much RAM | Set `frame_load_cap`, `select_every_kth`, or `force_rate`. |
| Result larger than target | Long video + tiny target hit the 32 kbps floor — reduce fps/resolution, disable audio, or raise the target. |
| No DOM bar / no preview | Hard-refresh the browser (Ctrl+F5) after installing/updating so the JS reloads. |
| Old workflow lost its connection | v2 uses `STAR_FILENAMES` (standalone). Use the included loader or `video_path` instead of VHS nodes. |
| `WinError 10054` in console | Unrelated Windows websocket noise from the ComfyUI server — harmless, ignore it. |

## 8. Releasing / shipping ffmpeg

The node needs an ffmpeg binary. Resolution order at runtime:

1. **Bundled binary** inside the node package (optional):
   ```
   StarVideoCompressor/bin/win/ffmpeg.exe     (Windows)
   StarVideoCompressor/bin/linux/ffmpeg       (Linux)
   StarVideoCompressor/bin/mac/ffmpeg         (macOS)
   ```
   If you want zero-install for users, drop static builds there. Sources
   for static builds: gyan.dev (Windows), johnvansickle.com (Linux),
   evermeet.cx (macOS). Keep them out of git (GitHub dislikes >50 MB files);
   attach them to the release or let users download them. Note that ffmpeg
   binaries containing libx264/libx265 are **GPL** — redistributing them
   means you must also provide the license texts and a source-code offer.
2. **`imageio-ffmpeg`** (recommended, in `requirements.txt`): ComfyUI
   Manager installs it automatically with the node; it bundles per-platform
   ffmpeg binaries as a normal pip package — no licensing or size headaches
   in your repo, works on Windows/Linux/macOS. This is what Video Helper
   Suite does too.
3. **System ffmpeg on `PATH`** as the final fallback.

## 9. Technical notes

- ffmpeg via `imageio-ffmpeg` (bundled build has libx264/libx265) or system
  ffmpeg; VP9/AV1 depend on the build. No ffprobe needed (all probing is
  parsed from `ffmpeg -i`).
- `+faststart` on MP4 outputs; pixel format forced to `yuv420p`.
- Progress is parsed from ffmpeg's `-progress` stream; DOM bar events are
  pushed over the ComfyUI websocket (`star_nodes.progress`).
- External audio is converted to a temporary 16-bit WAV before muxing;
  loader audio is extracted as WAV and returned as a core AUDIO dict.
- Filenames are sanitized (`..` stripped); output always stays inside the
  ComfyUI output/temp directory. Two-pass logs are cleaned automatically.
