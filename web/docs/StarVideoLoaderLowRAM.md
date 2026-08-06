# ⭐ Star Video Loader (Low RAM)

## Overview

The **Star Video Loader (Low RAM)** loads only a *window* of a video into memory, so even very long or high-resolution videos work on machines where the regular **Star Video Loader** (or the native ComfyUI loader) runs out of RAM.

Two working modes:

- **Direct seek mode** (default): ffmpeg fast-seeks to `start_frame` and decodes exactly `frame_count` frames. Only that chunk ever uses RAM.
- **Disk cache mode** (`cache_to_disk`): the video is decoded **once** into a raw frame cache inside the ComfyUI temp folder. Every later run — any window — reads the frames straight from disk via memory-mapping, without decoding the file again.

RAM usage is roughly `frame_count × width × height × 3 × 4` bytes. Example: 81 frames of 1080p ≈ 500 MB (vs ~22 GB per minute if the whole video were loaded).

## Inputs

### Required

- **video**: Video file from the ComfyUI input folder (upload supported)
- **start_frame** (INT): First frame of the window, counted in the *output* frame sequence (after `force_rate` / `select_every_kth`). Feed `next_start_frame` back into this to walk through the video in chunks
- **frame_count** (INT): How many frames to load — the only part of the video that uses RAM
- **force_rate** (FLOAT): Resample to this fps. 0 = keep the video's original frame rate
- **select_every_kth** (INT): Keep only every k-th frame (1 = keep all). Also divides the output fps
- **load_audio** (BOOLEAN): Extract the audio matching the loaded frame window
- **cache_to_disk** (BOOLEAN): Build/reuse the disk cache (see below)

## Outputs

- **images** (IMAGE): The loaded frame window
- **audio** (AUDIO): Audio matching the window's time range (or None if the video has no audio / `load_audio` is off)
- **fps** (FLOAT): Effective fps of the returned images (after force_rate / every-kth)
- **frames** (INT): Number of frames actually loaded (less than `frame_count` near the end of the video)
- **total_frames** (INT): Total frames in the video (exact in cache mode, estimated from duration in direct seek mode)
- **next_start_frame** (INT): `start_frame + frames` — connect or paste this into `start_frame` for the next chunk. When it reaches `total_frames`, you're done
- **info** (STRING): Report (window range, mode, cache location, audio, path)

## How It Works

### Direct seek mode
1. `start_frame` is converted to a timestamp (`start_frame / fps`) and ffmpeg seeks there directly — the beginning of the file is never decoded
2. Exactly `frame_count` frames are decoded into memory
3. Audio is extracted for the same time range (`-ss` / `-t`)

### Disk cache mode
1. On first run the whole video is decoded **streaming to disk** (memory stays flat — frames never pile up in RAM)
2. Frames are stored as a raw file (`frames.raw`) plus `meta.json` and `audio.wav` in `ComfyUI/temp/star_lowram_cache/<hash>/`
3. Later runs memory-map the raw file and copy out only the requested window — instant and low RAM
4. The cache is keyed by file path + modification time + size + fps settings, so replacing the video or changing `force_rate`/`select_every_kth` automatically builds a fresh cache

## Usage Example

### Chunked processing workflow
```
[Star Video Loader (Low RAM)]  start_frame=0,    frame_count=81
        │ images ──> [your processing] ──> [save/join chunk 1]
        └ next_start_frame=81

[Star Video Loader (Low RAM)]  start_frame=81,   frame_count=81
        │ images ──> [your processing] ──> [save/join chunk 2]
        └ next_start_frame=162
        ...
```

- Right-click `start_frame` → *convert to input* to drive it from another node (e.g. a counter/loop node)
- Combine the chunks with **⭐ Star Video Joiner**, or feed chunks into **⭐ Star Video Compressor** one at a time

### Common Use Cases

1. **Long video → image workflow**: process hour-long videos in 81-frame chunks without OOM
2. **Video-to-video**: extract a window, run it through a sampler, save, advance `start_frame`, repeat
3. **Preview scrubbing**: jump to any position instantly with `start_frame` (with cache mode, every scrub is instant after the first build)
4. **Frame extraction pipelines**: pair with ⭐ Star Frame From Video to pick exact frames from a window

## Tips

- **Chunk size**: pick `frame_count` so `frame_count × W × H × 12 bytes` fits comfortably in your free RAM (81 frames of 1080p ≈ 500 MB; 4K ≈ 2 GB)
- **Cache mode shines on re-runs**: the first build takes as long as decoding the video once; every run after that is near-instant
- **Cache cleanup**: cached folders live in `ComfyUI/temp/star_lowram_cache/` — safe to delete anytime; they rebuild on demand
- **Frame alignment**: with `select_every_kth` > 1, direct seek mode can be off by a few frames relative to a full decode (the k-th counting restarts at the seek point). Cache mode is always exact, since it slices the fully decoded sequence
- **Audio** is sliced to the exact time window of the loaded frames, so chunk audio stays in sync

## Category

Located in: **⭐StarNodes/Video**

## Technical Details

- Windowed decode: ffmpeg input seeking (`-ss` before `-i`) + `-frames:v N`, piped as raw rgb24
- Cache decode streams ffmpeg output directly to a file — memory usage stays flat regardless of video length
- Cache reads use `numpy.memmap`, so only the requested window is paged into RAM
- Shares the ffmpeg/probe/progress helpers with the other video nodes (`video_tools/star_nodes_common.py`)
- Shows the StarNodes DOM progress bar while decoding / building the cache
