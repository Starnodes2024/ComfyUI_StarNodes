# ⭐ Star Regional Spatial LoRA Canvas

Interactive canvas for regional generation: draw bounding boxes on a virtual canvas,
give every box its own short prompt plus an optional LoRA with its own strength.
Each LoRA only acts inside its own box, the rest of the picture stays untouched.

Category: `⭐StarNodes/Helpers And Tools`

## Wiring

Connect **all** outputs to the sampler: `model`, `positive`, `negative` and `latent`.
The `model` output carries the regional LoRAs - a sampler fed with the original
model renders the regions without their LoRAs.

Best paired with **⭐ Star Regional Sampler** (two-pass, see its help page), but any
sampler works (KSampler, SamplerCustomAdvanced, ...).

## Inputs

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Base diffusion model (quantized int8 / nvfp4 / fp8 checkpoints are fine) |
| `clip` | CLIP | Text encoder for the main, regional and negative prompts |
| `ratio` + `custom_ratio` | DROPDOWN/STRING | Same ratio selection as ⭐ Star Advanced Ratio/Latent (custom accepts e.g. `21:9`) |
| `resolution` | DROPDOWN | Named size presets or `custom` |
| `custom_width` / `custom_height` | INT | Used when `resolution` is `custom` (rounded to multiples of 16) |
| `latent_channels` | DROPDOWN | `4 (SD/SDXL)` / `16 (FLUX/QWEN/ZIT)` → 8× latent, `128 (FLUX2)` → 16× latent. **Default: 128 (FLUX2)** |
| `ratio_from_image` | BOOLEAN | Take the aspect from the connected image instead of the ratio widgets |
| `image` (optional) | IMAGE | Source image for `ratio_from_image` |
| `batch_size` | INT | Batch size of the output latent |
| `main_prompt` | STRING | Scene prompt. Describe the whole picture including roughly who/what is where |
| `refine_prompts` / `refine_scope` / `refiner_clip` / `refiner_seed` | | Optional prompt rewrite with an instruct text encoder (embedded regional system prompts) |
| `mask_feather` | FLOAT | Soft edge of the regions in latent pixels (default 2.0). Box sides on the canvas border stay hard |
| `region_prompt_mode` | DROPDOWN | `scene-aware` (default): region prompt + position hint + main prompt. `region only`: box prompt as written |
| `global_blend` | FLOAT | Share of the main prompt (without LoRA) kept inside the boxes. 0 = regions fully own their box |
| `negative_prompt` | STRING | Only used with CFG > 1; mirrored into every region with that region's LoRA |
| `regions_json` | STRING (hidden) | Serialized boxes from the canvas editor. Do not edit by hand |

## Outputs

| Output | Type | Description |
|---|---|---|
| `model` | MODEL | Model with the regional LoRAs attached. **Use this one for sampling** |
| `positive` | CONDITIONING | Main prompt outside the boxes + one entry per box |
| `negative` | CONDITIONING | Negative prompt with the same regions and LoRAs |
| `region_masks` | MASK | Final region weights at latent resolution (`[N, h, w]`), for preview or compositing |
| `latent` | LATENT | Empty latent in the computed size and channel format |

## Canvas usage

1. Pick `ratio` / `resolution` / `latent_channels` on the node.
2. Click **🎨 Open Spatial Canvas**.
3. Draw boxes by click-drag. Move by dragging inside a box, resize via the corner dot.
4. Select a box to edit its prompt, LoRA and LoRA strength.
   - LoRA `none` = prompt-only region.
   - Empty prompt + LoRA = the LoRA is applied to that area of the main scene.
   - No prompt and no LoRA = box is skipped.
   - **Face detailer (YOLO)** marks this box for an optional face-refinement pass
     after regional sampling. An `F` badge appears on the box. Connect CLIP + VAE
     on ⭐ Star Regional Sampler to enable it; the box's character LoRA stays
     active during its face pass.
5. **Apply** stores all boxes in the workflow.

## How it works

- Every region is one extra **full-canvas** model evaluation per step, with the
  region's prompt and LoRA. The region still sees the whole picture, so lighting,
  perspective and style stay consistent, and only the part inside the box is kept.
- The region weights and the weight of the main prompt add up to exactly 1.0 on
  every latent pixel. Inside a box only the region's result is used, so the LoRA
  arrives at full strength and nothing is averaged into mush. Overlapping boxes
  share their overlap evenly.
- LoRAs never modify the model weights. They run as a small side path inside the
  patched layers and are switched on only while their own region is evaluated.
  This is why quantized checkpoints work without requantizing or weight swapping,
  and why no ComfyUI core file is involved.
- With CFG > 1 the negative is evaluated per region with the same LoRA, so the
  guidance never compares a LoRA model against the plain model.

## Tips for sharp results

- Use a resolution the model is trained for (about 1-2 MP). Get more pixels from
  the ⭐ Star Regional Sampler detail pass instead of a huge first pass.
- Give regional renders a few more steps than usual: 8-12 on turbo/distilled
  models, 25-35 on base models.
- Write the main prompt as the complete scene ("two people standing next to each
  other in a street at night ...") and the box prompts as the individual subjects.
- Make boxes generous - a box should contain the whole subject with some margin.
- Cost: one extra model evaluation per box and step (two with CFG > 1).
- Diffusers-format LoRAs (`transformer_blocks.N.attn.to_q/to_k/to_v`) are
  supported - their slices of fused qkv weights are applied inside the box only.
- LoRA files that contain full weight diffs can only be
  applied partly; the console reports how many entries were skipped.

## model_preset

Recommended per-model setup for the three supported models:

| Preset | Latent format | Feather | Sampler recipe |
|---|---|---|---|
| MOSS Image | 128ch (16x) | 2.0 | 30 steps, cfg 4.5, refine 12 @ 0.3 |
| Krea2 | 16ch (8x) | 4.0 | 8 steps, cfg 1.0, refine 4 @ 0.35 |
| Qwen Image 2.1 | **64ch** (16x) | 2.0 | 10 steps, cfg 1.0, refine 5 @ 0.35 |

The preset corrects `latent_channels` automatically (a wrong format breaks the
latent *and* the masks) and uses its feather when `mask_feather` is left at the
default 2.0. `auto` uses the widgets as set.

Note: Qwen Image 2.1 latents are 64-channel - pick `64 (QWEN2.1)` manually or
use the preset.

## Troubleshooting

Important: the `model` output must feed the sampler - the regional LoRAs only
travel through it. Regional prompts work even when it is not connected, so a
wrong wire fails silently except for the warnings below.

The console tells you what happened per box:

- `+ LoRA '...' @ s on N layers` - the LoRA was attached; N should be large
  (e.g. ~250 on MOSS Image).
- `LoRA '...' matches no layer of this model` - key format not compatible with
  the checkpoint.
- `N LoRA entries are not regional-capable` - weight-diff/set entries that have
  no forward-pass form were skipped.
- In the sampler: `the conditioning carries region LoRAs but the model has
  none` - connect the canvas `model` output, not the checkpoint loader's.
- In the sampler: `regional LoRAs were registered but never fired` - the token
  never reached the model (conditioning lost its hooks) or the patched layers
  were never called.

Set `STAR_REGIONAL_DEBUG=1` for per-eval logging of the region token and
injection runs.

Note on dynamic VRAM (aimdo): ComfyUI cannot run hooked conditioning on a
dynamic model patcher and swaps in a rebuilt non-dynamic delegate model
(`samplers.py` `inner_set_conds`). Because the regional LoRA side path attaches
to the model's modules, the node clones the model non-dynamic at build time
whenever a box uses a LoRA. Prompt-only workflows keep dynamic VRAM.
