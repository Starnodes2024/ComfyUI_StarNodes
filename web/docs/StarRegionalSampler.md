# ⭐ Star Regional Sampler

Two-pass sampler for ⭐ Star Regional Spatial LoRA Canvas (works with any conditioning).

Category: `⭐StarNodes/Sampler`

1. **Main pass** - builds the picture from the regional conditioning.
2. **Detail pass** - adds a little noise back and samples again with the same
   regions and LoRAs. All regions already agree on one picture at this point, so
   this pass only resolves soft transitions and fine detail. Optionally the latent
   is upscaled first for a higher output resolution.

## Inputs

| Input | Description |
|---|---|
| `model` / `positive` / `negative` / `latent` | Connect the outputs of the canvas node |
| `seed` | Seed of the main pass (detail pass uses seed + 1) |
| `steps` | Main pass steps. 8-12 for turbo/distilled models, 25-35 for base models |
| `cfg` | 1.0 for turbo/distilled and guidance-embedded models |
| `sampler_name` / `scheduler` | As in KSampler |
| `denoise` | Main pass denoise |
| `refine_steps` | Detail pass steps, 0 = off |
| `refine_denoise` | 0.25-0.4 keeps the composition and sharpens; 0.45-0.6 when upscaling |
| `refine_upscale` | Latent upscale before the detail pass, 1.0 = off |

## Recommended settings

All sampling values are set on the widgets - nothing overrides them. Starting
points per model:

| Model | steps | cfg | refine_steps | refine_denoise | refine_upscale |
|---|---|---|---|---|---|
| MOSS Image | 30 | 4.5 | 12 | 0.30 | 1.0 |
| Krea2 | 8 | 1.0 | 4 | 0.35 | 1.0 |
| Qwen Image 2.1 | 10 | 1.0 | 5 | 0.35 | 1.0 |
| any model, crisp + larger | +0 | same | +2 | 0.5 | 1.5 |

## sigmas input (optional)

Custom sigma schedule, same contract as SamplerCustomAdvanced - connect any
SIGMAS source (e.g. the ⭐ Star custom sigmas node, SplitSigmas, RES4LYF
schedules). When connected it overrides `steps`, `scheduler` and `denoise` for
the main pass; the detail pass reuses the same schedule scaled by
`refine_denoise`.

## Per-box face detailer (optional)

Mark **Face detailer (WeVisOne)** on any box in the canvas editor and connect the
sampler's optional `clip` and `vae` inputs. After the normal regional passes,
WeVisOne-VL scans the **whole image once**, then assigns detected faces to the
nearest marked boxes one-to-one, each with its own precise alpha matte. The
detected face bounds - not the original canvas rectangle - positions the
refinement. The soft matte is feathered and edge-faded for pasting, so neither
the canvas box nor a hard detector rectangle leaves a visible frame. Faces
outside all marked boxes are ignored.
The box's prompt and character LoRA are reused for this pass.

- `detector_model`: WeVisOne-VL encoder from `models/text_encoders`:
  - `wevisone_vl_2b_int8_convrot.safetensors` (2.7 GB, default, fastest)
  - `wevisone_vl_4b_int8_convrot.safetensors` (4.9 GB, better quality)
  - `wevisone_vl_4b_bf16.safetensors` (9.0 GB, best quality)
  A missing encoder is downloaded automatically from `Comfy-Org/WeVisOne-VL` on first use.
- `face_steps`: 12 by default; 0 disables all face detailing.
- `face_denoise`: 0.5 by default.
- Uses the main sampler's cfg, sampler and scheduler. No face found = box skipped.
- Without CLIP + VAE a warning is logged and the regional result is unchanged.

The second output **`image`** contains the exact face-refined pixels shown in the
preview. Use it with Save Image / Preview Image. The first output `latent` is the
same image re-encoded through the VAE for further latent processing; decoding it
again can differ slightly. Without a VAE connection the `image` output is `None`.

## Standalone ⭐ Star Face Detailer

The same engine is available as a separate node for any image. Connect `image`,
`model`, `clip` and `vae`, enter a prompt, and sample. WeVisOne-VL detects and
refines all faces in the image with per-face alpha mattes, with 12 steps /
denoise 0.5 / guide size 1024 by default. `crop_factor` and `feather` are
optional advanced widgets. The detector is the same selectable WeVisOne-VL
encoder as above (2B int8 default, auto-downloaded on first use) - no
Impact-pack, ultralytics or provider nodes required. See `StarFaceDetailer.md`
for all features (refine targets, prompt refiner, per-face prompts/LoRAs,
color match and reference-image swap).

### Reference-image face swap

The standalone node's optional `reference_image` enables identity conditioning:

- **Fibo2:** the reference is center-cropped/resized to the guide size, passed to
  the text encoder, and its VAE latent is attached to both conditionings - the
  same mechanism as TextEncodeFibo2.
- **MOSS Image Edit:** uses the native ai-toolkit reference encoding path.
- **Qwen Image 2.1:** uses TextEncodeQwenImage21's native text-encoder images and
  VAE reference latents. Its RGBA VAE output is converted back to the input's
  channel count automatically.
- Other model families raise a clear error rather than ignoring the reference.

For replacement use a prompt such as `replace the face with the person in the
reference image, preserve pose, hair, lighting and background`, and denoise
**0.75-1.0**. Lower denoise values are better for ordinary face enhancement.

## preview input (optional)

Connect the ⭐ Star Preview node (comfyui_starnodes pack) for a live sampling
preview, same as on ⭐ Star Sampler Unified. Works for both passes.

## Warnings

- `the conditioning carries region LoRAs but the model has none` - the canvas
  node's `model` output is not connected (regional prompts still work, LoRAs
  would not).
- `regional LoRAs were registered but never fired` - the region token never
  reached the model or the patched layers were never called; enable
  `STAR_REGIONAL_DEBUG=1` for details.
