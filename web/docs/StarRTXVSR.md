# ⭐ Star Advanced RTX VSR — Help

AI upscaling with **NVIDIA RTX Video Super Resolution** (NVIDIA VFX SDK) —
the same technology as the RTX VSR video player enhancement, as a ComfyUI
node. Works on plain images and video frame batches alike.

By default every frame is super-resolved to **4K first**, then resized down
with a high-quality lanczos filter when a lower output size is chosen — so
2K and Full HD outputs still carry the full VSR detail treatment. **8K**
renders directly at 8K (the VFX SDK supports it natively).

Disable **render_at_4k** to let VSR upscale directly to the chosen output
size instead — lower VRAM, faster, but less detail on small outputs.

Self-contained in the StarNodes pack — the original `comfyui_nvidia_rtx_nodes`
pack is **not** required.

## Requirements

- An **NVIDIA RTX GPU** (VSR is an RTX-only hardware feature)
- The `nvidia-vfx` pip package (included in the pack's requirements.txt)

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `images` | IMAGE | frames/images to upscale — a video IMAGE batch works directly |
| **upscaled_images** out | IMAGE | the upscaled frames at the chosen output size |

## Widgets

- **output_size** — `2K` (default) / `8K` / `4K` / `Full HD`. Aspect ratio of
  the input is preserved (fit inside 7680×4320 / 3840×2160 / 2560×1440 /
  1920×1080, snapped to a multiple of 8).
- **strength** — VSR quality: `LOW` / `MEDIUM` / `HIGH` (default) / `ULTRA`.
  Higher is sharper and slower.
- **render_at_4k** — `True` (default): render the VSR pass at 4K first (8K
  renders directly), then resize down to the chosen output with a lanczos
  filter. Keeps full VSR detail at any output size. `False`: let VSR upscale
  directly to the chosen output — lower VRAM and faster, but less detail on
  small outputs (2K / Full HD). Ignored for 8K, which always renders natively.
- **batch_size** — frames processed per batch (default `2`, range 1–64).
  Lower to `1` on low-VRAM GPUs to avoid out-of-memory errors; raise on
  GPUs with headroom to speed up long clips.
- **downscale_filter** — filter used when downscaling the 4K/8K VSR result
  to 2K/Full HD. Only used when **render_at_4k** is on and the output size
  is below the VSR render size.
  - `lanczos` (default) — sharpest, preserves VSR detail best; slight
    ringing/halos on hard edges. Best default for video.
  - `bicubic` — balanced, moderate sharpening; more aliasing on diagonals.
  - `area` (default) — box-average, cleanest anti-aliasing for large
    downscale factors (e.g. 4K→Full HD), softest. Good for smooth video.
  - `nearest` — fastest, blocky. Not recommended for video.

## Notes

- With **render_at_4k** on, small inputs upscale net-even when you pick Full
  HD (e.g. a 1080p input is VSR-enhanced at 4K and resized back to 1080p) —
  this is intentional: you get the VSR sharpening without a resolution change.
- With **render_at_4k** off, a 1080p input picked at Full HD is essentially
  a no-op (VSR renders straight to 1080p). Pick a higher output_size or turn
  render_at_4k back on to get a visible effect.
- No CUDA-capable RTX GPU → the node stops with a clear error.
