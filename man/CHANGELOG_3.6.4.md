# StarNodes 3.6.4 — Update Changelog

## New Nodes

### ⭐ Star Model Loader (`StarModelLoader`) + ⭐ Star Model Loader Advanced
One loader to rule them all: checkpoint or split diffusion model, up to 2
text encoders, up to 3 VAEs, sampler/scheduler/steps — plus JSON presets
(`json/star_model_loader_presets.json`). The Advanced variant adds
`latent`, `width`, `height` outputs with the MOSS/Krea2/Qwen/Flux2
latent-channel presets.

- **Stable fixed outputs** — the declared sockets never change shape, so
  saved workflows survive every frontend update; changing a dropdown no
  longer drops your wires. Unselected outputs simply emit `None`.
- **Filled vs. hollow sockets** — active outputs are filled, type-colored
  and labeled with the selection (`juggernaut_moss_turbo…`); unused ones
  are hollow dimmed rings, so you see at a glance what to wire.
- Sampler/scheduler outputs carry the real `KSampler` value lists, so they
  plug straight into any converted combo widget (StarSampler, KSampler, …).

Doc: `web/docs/StarModelLoader.md`

### ⭐ Star Universal Conditioner (`StarUniversalConditioner`)
One conditioning node for text + up to 10 reference images: vision-tower
encoding, VAE reference latents and negative-zero handling for modern
models (Flux2, Qwen-Image-Edit, MOSS, …). Static inputs `vae` +
`Image 1…10` — restored workflows keep every wire.

- `reference_resize` — `none / 0.5 MP / 1 MP / 1.5 MP / 2 MP / 3 MP`
  (default 1 MP), snapped to a 32-px grid **before** vision tokens and VAE
  encoding: big references stop eating VRAM and token budget.
- Fixed Qwen Image 2.1 reference conditioning.

Doc: `web/docs/StarUniversalConditioner.md`

### ⭐ Star LoRA Converter (`StarLoraConverter`)
Shrink an existing LoRA's rank with SVD and/or re-cast it to FP16 / FP8 —
saved as a new file in `models/loras/resized` (source never touched).

- `target_rank` `keep / 128…1` — modules already ≤ target are kept as-is;
  alpha is rescaled so the effective strength is preserved exactly.
- Kohya (`lora_down/lora_up`) & Diffusers (`lora_A/lora_B`) keys, Linear +
  Conv2d; everything else (norms, biases, embeddings, lokr/hada) passes
  through untouched.
- Exact QR-core factorization — **~100× faster** than a dense full SVD,
  works on CPU in seconds. Node progress bar + console progress.
- Metadata preserved (`ss_network_dim`/`alpha` updated), `_v2` naming on
  collision, human-readable `conversion_report` output.

Doc: `web/docs/StarLoraConverter.md`

### ⭐ Star Regional Spatial LoRA Canvas (`StarRegionalSpatialLoraCanvas`) + ⭐ Star Regional Sampler
Draw boxes on an interactive canvas — each box gets its own prompt, LoRA
and strength, rendered together with full-scene context. Regional LoRAs
run as a forward side path instead of rewriting weights, so quantized
checkpoints (int8 / nvfp4 / fp8) work too. MOSS Image, Krea2 and Qwen
Image 2.1 latent presets, optional custom sigmas, Star Preview support and
a second detail pass.

Docs: `web/docs/StarRegionalSpatialLoraCanvas.md`, `web/docs/StarRegionalSampler.md`

### ⭐ Star Face Detailer (`StarFaceDetailer`)
Detect → crop → refine → paste back, done properly:

- Native ComfyUI **WeVisOne-VL** segmentation (no YOLO dependency).
- Per-face prompt text inputs, per-face LoRA search/support, multi-face
  handling, skip-per-face and regional LoRA integration.
- Crop → fixed-size refine → resize-back → masked paste with seam-free
  blending; crop factors below 1.0 for tight bust/portrait refinement.
- Arbitrary-target refinement (not only faces).

Docs: `web/docs/StarFaceDetailer.md`, `web/docs/StarMultiFaceDetails.md`

### ⭐ Star Minimax Remake Option (`StarMinimaxRemakeOption`)
Viggle-Remake mode for ⭐ Star Minimax All In One — **no Viggle node pack
needed**: pick the swap/edit (or lowrank) Viggle-Remake LoRA, feed the
source clip's 24 fps frames + soundtrack from ⭐ Star Video Loader and
connect 1–4 painted reference frames. Pins pin a reference as a keyframe
on the timeline (released below `keyframe_until` sigma); the source audio
is held clean through the whole denoise.

Doc: `web/docs/StarMinimaxRemakeOption.md`

### ⭐ Star Model Merge (`StarModelMerge`) + ⭐ Star Model AIO Merge (`StarModelAIOMerge`)
Merge two checkpoints/diffusion models (or AIO files incl. text encoder +
VAE) with ratio control — saved as a new safetensors file.

### ⭐ Star GPU Cooldown (`StarGpuCooldown`)
Insert between heavy nodes: offloads/clears VRAM and lets the GPU settle
so the next sampler starts with full memory.

## Improvements & Changes

### ⭐ Star Video Loader — crop, preview & presets
- **Crop dialog with live preview**: load the video, draw the crop rect on
  the real frames — backend crops exactly what you drew; crops can be
  deleted again. No workflow run needed.
- Resize presets + `seconds` output + optional `minimax_snap` (snaps
  frames/duration to MiniMax-friendly values).
- The crop overlay frontend (`web/js/star_video_compressor.js`) got a full
  styling/stage upgrade.

### ⭐ Star Minimax All In One
- `remake_settings` input for the new Remake Option (see above).
- **17n+5 frame handling**: the required MiniMax frame count is achieved
  by pad-render-trim — your requested source frames and continuous audio
  are preserved exactly.
- Reference-image megapixel caps: `none / 0.5 / 1.0 / 1.5 MP`
  (default 1.0 MP) to keep VRAM and ref tokens in check.
- Reverted to the proven Viggle-style all-keyframe release behavior after
  CUDA allocator failures in the RefMod guide variant.

### ⭐ StarSampler & ⭐ Star SD Upscale Refiner (Advanced)
- Further CUDA-stability hardening: GPU-fatal errors propagate so ComfyUI
  can unload models and recover instead of crashing the driver.
- Upscale refiner: crop → fixed-size refinement → resize-back → masked
  paste pipeline, crop factors < 1.0, seam fixes.

### ⭐ Star Preview — Flux2 Tiny VAE
Preview decoding for Flux2 latents via `fal/FLUX.2-Tiny-AutoEncoder`
("taef2", 128 ch) — the extra_decoder + residual block + diffusers
AutoencoderTiny stack runs in-process like the other TAE previews.

### ⭐ Star Custom Sigmas
New baked preset **"Viggle Reference 6 Steps"** matching the Viggle Remake
reference schedule.

### ⭐ StarText
New system-prompt preset **"Minimax Recreate Option"** — the full
six-section video-editing / character-replacement prompt template
(`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`) for
`<Video 1>` + `<Picture N>` reference workflows.

### Small node fixes
- `⭐ Star Advanced Ratio/Latent` — new **64 ch (QWEN2.1)** latent option;
  latent-space division now applies to all ≥64-channel latents.
- `⭐ Star Lucida RMBG` — bf16 selection now falls back to fp16 with a
  console note (torchvision's `deform_conv2d` has no bf16 kernel).
- `⭐ Star Realistic Film Grain` — extra channels (e.g. RGBA alpha) are
  preserved instead of being mangled by the grain math.

## Removed
- The experimental face-swapper node was removed (the WeVisOne-VL detailer
  pipeline covers the use case more reliably).

## New Wildcards
- `Dataset-Prompt_Portraits.txt` — 50 unisex photorealistic close-up
  portrait captions for LoRA training datasets.
- `Dataset-Leon-Prompts.txt` — 100 dataset captions for a male character
  ("Leon"): tactical/cop outfits, casual/formal/sexy clothing, action
  poses, portraits, close-ups, angles, shirtless & tasteful figure
  studies — variable detail in the caption, `man` as trigger placeholder.

## Changed Files (this update)
- `__init__.py`, `pyproject.toml`, `README.md` — registrations + 3.6.4
- `misc/star_model_loader.py`, `web/js/star_model_loader.js`,
  `json/star_model_loader_presets.json` (new)
- `text_io/star_universal_conditioner.py` (new)
- `misc/star_lora_converter.py` (new)
- `misc/star_model_merge.py`, `misc/star_model_aio_merge.py` (new)
- `misc/star_gpu_cooldown.py` (new)
- `samplers/star_regional_spatial_lora_canvas.py`,
  `samplers/star_regional_sampler.py`, `samplers/star_face_detailer.py`,
  `web/js/star_regional_spatial_lora_canvas.js` (new)
- `minimax/star_minimax_remake_option.py` (new),
  `minimax/star_minimax_aio.py`, `minimax/minimax_common.py`,
  `minimax/__init__.py`
- `video_tools/star_video_loader.py`, `web/js/star_video_compressor.js`
- `samplers/star_sampler.py`, `samplers/star_custom_sigmas.py`,
  `web/js/star_custom_sigmas.js`
- `image_tools/star_sd_upscale_refiner.py`,
  `image_tools/star_latent_ratio_megapixel.py`,
  `image_tools/star_lucida_rmbg.py`,
  `image_tools/star_realistic_film_grain.py`
- `misc/star_preview.py`
- `json/startext.json`
- `web/index.js`, `web/docs/*` (many new/updated help pages)
- `wildcards/Dataset-Prompt_Portraits.txt`,
  `wildcards/Dataset-Leon-Prompts.txt` (new)

---

## How to update

**ComfyUI Manager (easiest):** Manager → *Update All* (or search
"StarNodes" → Update), then restart ComfyUI.

**git pull (existing install):**
```bash
cd ComfyUI/custom_nodes/ComfyUI_StarNodes
git pull
```
then restart ComfyUI.

**Fresh install (git clone):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes
```
then restart ComfyUI (deps: `pip install -r requirements.txt` if needed).

Repo: https://github.com/Starnodes2024/ComfyUI_StarNodes

After updating, **restart ComfyUI** and do a hard refresh (**Ctrl+F5**)
so backend registrations and frontend JS reload. Workflows saved with the
old dynamic model-loader sockets may need their loader outputs rewired
once — from then on the fixed sockets stay stable.
