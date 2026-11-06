# StarNodes 3.5.0 — Update Changelog

## New Nodes

### ⭐ Star Tiled RelayVSR Upscaler (`StarTiledRelayVSRUpscaler`)
Image & video upscaling with the RelayVSR diffusion model — fully self-contained
(conditioning, one-step sampling and VSR flash decoding are inlined; no native
core nodes loaded, so it keeps working after core changes).

- `image` — input at native resolution; the checkpoint decides the factor (x2/x4)
- `model_name` / `vae_name` — dropdowns, or `model_override` for patched models
- `seed` — default 0, control set to `fixed`
- `tiled` — toggle (default on). Off = whole image in a single pass
- `max_tile_size` — auto tile grid, default 1024 (works with very large images)
- `tile_overlap` — seam-blend overlap (default 0.25)
- `detail` — slider 0.0–1.0, default 0.6 (scales the model residual over the bicubic base)
- `batch_size` — 0 = auto (whole input as one video pass, fastest for videos), 1 = each image separately, >1 = fixed chunk size

Doc: `web/docs/StarTiledRelayVSRUpscaler.md`

## Bug Fixes

### Fixed: CUDA driver crash ("CUDA unknown error" → reboot required)
- `samplers/star_sampler.py` — the ZIT two-pass and Split-sampler options wrapped
  the whole GPU sampling in `except Exception`, swallowed OOM/CUDA errors and
  immediately ran another full sampling pass against a failed CUDA state.
  GPU-fatal errors now re-raise (`rethrow_gpu_fatal`) so ComfyUI's executor can
  unload models and recover, like the native KSampler.
- `image_tools/star_sd_upscale_refiner.py` — the tiled-diffusion wrapper answered
  an interrupt request by silently returning the noisy input (`return x_in`) and
  letting sampling continue at full GPU load. It now raises
  `InterruptProcessingException` so the run actually stops.

### Fixed: Star Video Compressor OOM on large IMAGE batches
- `video_tools/star_video_compressor.py` — `_to_uint8_frames` triplicated huge
  batches in RAM (51 GB source → ~127 GB peak). The batch is now streamed to a
  temp rawvideo file on disk in slices and ffmpeg reads the file directly —
  near-zero extra RAM. drop_first/drop_last and odd-size padding still work.

## New Wildcards
- `metalobjects.txt` — 500 short prompts: objects/statues/animals/scenes made of
  various metals (liquid chrome, brushed silver, shiny gold, liquid copper, …)
- `liquid-chrome-metals.txt` — 500 entries, exclusively shiny liquid/mirror
  chrome textures in gold/silver/platinum/copper tones, smooth + structured

## Changed Files (this update)
- `__init__.py` — register StarTiledRelayVSRUpscaler, version 3.5.0
- `pyproject.toml`, `README.md` — version 3.5.0
- `samplers/star_sampler.py`
- `image_tools/star_sd_upscale_refiner.py`
- `image_tools/star_tiled_relayvsr_upscaler.py` (new)
- `video_tools/star_video_compressor.py`
- `web/docs/StarTiledRelayVSRUpscaler.md` (new)
- `wildcards/metalobjects.txt` (new)
- `wildcards/liquid-chrome-metals.txt` (new)
