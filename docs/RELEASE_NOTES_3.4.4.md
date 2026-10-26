# Release Notes - Version 3.4.4

## Summary
This release is all about LoRA dataset workflows: images and caption text files now travel together through the whole pipeline. ⭐ Star Save Image+ writes `.txt` sidecars, ⭐ Star Load Image+ and ⭐ Star Image Loader 1by1 read them back, and two new wildcard prompt packs generate dataset-ready captions. Also: external conditioning + a whole-latent toggle for the SD Upscale Refiner, and an important internal sampling cleanup.

## Changes

### New Features

#### ⭐ Star Save Image+ (`StarSaveImagePlus`)
- **Optional `caption` STRING input** — connect any text (e.g. a prompt or wildcard output) and the node saves a `.txt` file with the **exact same filename** as the image in the **same selected folder**
- Empty/unconnected captions write nothing; preview mode writes nothing
- **New `Datasets` folder preset** for quick dataset output

#### ⭐ Star Load Image+ (`StarLoadImagePlus`)
- **New `load with txt` toggle** + **`caption` STRING output** — loads the `.txt` sidecar matching the selected image so captions are never lost
- Change detection watches the `.txt` file too — editing the caption on disk refreshes the output

#### ⭐ Star Image Loader 1by1 (`StarImageLoader1by1`)
- **New `caption` STRING output** — automatically loads the `.txt` sidecar of each sequentially loaded image (appended output slot, existing workflows unaffected)

#### New Wildcards
- `wildcards/Dataset-Prompts-Female.txt` / `wildcards/Dataset-Prompts-Male.txt` — 200 unique entries each (100 dressed / 100 nude) with varied framings, angles, poses, expressions and lighting; body and face details stay neutral so they work with any character

### Enhancements

#### ⭐ Star SD Upscale Refiner Advanced (`StarSDUpscaleRefinerAdvanced`)
- **Optional `pos_condition` / `neg_condition` CONDITIONING inputs** — when connected, external conditioning is used instead of the internal prompt fields (e.g. wire regional prompters, style conditionings or a different CLIP)
- **New `refine_tiled` toggle** (default on) — turn off to refine the whole upscaled latent in one pass (seam-free and faster, but needs much more VRAM on large images)

#### ⭐ Star Sexy Builder (`StarSexyBuilder`)
- **New `hairstyle` and `hair_color` dropdowns** with auto-combining (`long wavy` + `blonde` → `long wavy blonde hair`)

### Bug Fixes & Internal

- **Removed the global `sampling_function` monkeypatch** from Star SD Upscale Refiner Advanced — Auto-CFG now relies on the native `sampler_cfg_function` hook and can no longer break other samplers or future ComfyUI versions
- **Sigma-path sampling output fix** in ⭐ Star Sampler, ⭐ Star Flux2/Qwen Inpainter and ⭐ Star Qwen Outpainter — the nodes now use the main sampler output instead of `denoised_output` (matters for partial schedules)
- **New shared helper** `samplers/star_sampler_utils.py` — centralizes sigma-based sampling on the stable core API (`comfy.samplers.CFGGuider`)

### Updated Files

#### Core Files
- `__init__.py` - Version 3.4.4
- `README.md` - Version 3.4.4 badge
- `pyproject.toml` - Version 3.4.4

#### Image Tools
- `image_tools/star_save_image_plus.py` - caption sidecar saving
- `image_tools/star_load_image_plus.py` - load_with_txt toggle + caption output
- `image_tools/star_image_loader_1by1.py` - caption output
- `image_tools/star_sd_upscale_refiner.py` - external conditioning inputs, refine_tiled toggle, monkeypatch removed
- `image_tools/presets.json` - Datasets preset

#### Samplers & Misc
- `samplers/star_sampler.py` - shared sampling helper, output index fix
- `samplers/star_sampler_utils.py` - new shared sigma-path helper
- `misc/StarFlux2Inpainter.py` - shared sampling helper
- `qwen/star_qwen_outpainter.py` - shared sampling helper

#### Text & Prompts
- `text_io/star_sexy_builder.py` - hairstyle + hair_color
- `json/star_sexy_builder.json` - new option lists
- `wildcards/Dataset-Prompts-Female.txt` - new
- `wildcards/Dataset-Prompts-Male.txt` - new

#### Web Assets
- `web/docs/StarSaveImagePlus.md` - caption input + Datasets preset
- `web/docs/StarLoadImagePlus.md` - load_with_txt + caption output
- `web/docs/StarImageLoader1by1.md` - caption output
- `web/docs/StarSexyBuilder.md` - hair selectors

---

## Version 3.4.3 (2026-10-25) Recap
- ⭐ Star Horror Villains Builder + ⭐ Star Sexy Builder added to the builder family
- ⭐ Star CLIP Prompt Refiner: `max_length` default 512 → 2048
- ⭐ Star Prompt Preview And Edit: preview size persistence fix after workflow tab switches
