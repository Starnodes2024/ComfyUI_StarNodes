# ⭐ Star Model Loader

One node that loads a whole stack of models — a checkpoint, up to three
diffusion models, two text encoders and three VAEs — with a fixed set of
**output sockets**, so a single node can feed an entire workflow. Save the
current selection as a named **preset** and re-apply it with one click later.

## Widgets

| Widget | Type | What it does |
|---|---|---|
| `Preset` | combo | `(none)` or a saved preset. Picking a preset fills the model dropdowns with its stored selections |
| `Checkpoint` | combo | a file from `models/checkpoints` — creates **three** outputs: `checkpoint_model` (MODEL), `checkpoint_clip` (CLIP), `checkpoint_vae` (VAE) |
| `Model 1–3` | combo | files from `models/diffusion_models` — one `model_N` (MODEL) output each |
| `Text-Encoder 1–2` | combo | files from `models/text_encoders` — one `text_encoder_N` (CLIP) output each. The encoder type is auto-detected from the filename (`flux`, `sd3`, `wan`, `qwen`, `ltxv`, `hunyuan`, …) with `stable_diffusion` as fallback — the detected type is logged |
| `VAE 1–3` | combo | files from `models/vae` — one `vae_N` (VAE) output each |
| `Sampler` | combo | passes the sampler **name** through — the output carries the real `KSampler.SAMPLERS` list, so it wires straight into a `sampler_name` widget *converted to input* (works on the Star samplers, KSampler, KSamplerSelect, …) |
| `Scheduler` | combo | passes the scheduler **name** through — the output carries the real `KSampler.SCHEDULERS` list, wire it to a scheduler widget *converted to input* (right-click a widget → convert widget to input) |
| `Steps` | int | passes the step count through as an `INT` output — wire it to a steps input or converted widget; `0` = no output |

Every widget left at `(none)` (or `Steps = 0`) is skipped at load time — its
output returns `None`, so simply leave those sockets unconnected.

## Buttons

- **Save Preset** — asks for a name and stores the current selection of all
  nine dropdowns. If the name already exists you are asked to confirm the
  overwrite.
- **Update Preset** — overwrites the preset currently selected in the
  `Preset` dropdown with the current selection (asks for confirmation).
- **Delete Preset** — removes the preset selected in the `Preset` dropdown
  (asks for confirmation).

Presets are stored in `json/star_model_loader_presets.json` inside the
StarNodes folder, so they can be backed up or shared between installs.

## Outputs

The outputs are **fixed** — always present, with real types, in this order:

1. `checkpoint_model` (MODEL) / `checkpoint_clip` (CLIP) / `checkpoint_vae` (VAE)
2. `model_1`, `model_2`, `model_3` (MODEL)
3. `text_encoder_1`, `text_encoder_2` (CLIP)
4. `vae_1`, `vae_2`, `vae_3` (VAE)
5. `sampler` (sampler list), `scheduler` (scheduler list), `steps` (INT)

Sockets of unselected slots output `None` — leave them unconnected. Fixed
sockets keep workflow links stable across saves, reloads and tab switches.

**Visual guide:** outputs with an active selection are drawn as filled,
type-colored circles and their label shows the selection (e.g.
`juggernaut_moss_turbo…`, `sampler: lcm`, `steps: 8` — model/clip/vae
sockets are identified by color, so only `sampler`/`scheduler`/`steps` keep
a prefix). Unused outputs are hollow, dimmed rings with just their plain
name. This is purely cosmetic — socket names, types and order never change.

## Star Model Loader Advanced

`⭐ Star Model Loader Advanced` is the same node plus the **Star Advanced
Ratio/Latent** generator built in. It uses the **same presets file** — latent
settings are stored as extra keys in each preset, so a preset saved in the
Advanced node re-applies its latent settings there while the basic loader
simply ignores them (and vice versa: saving from the basic node keeps any
previously stored latent settings untouched).

Additional widgets (after `Steps`):

| Widget | Type | What it does |
|---|---|---|
| `Ratio` | combo | aspect ratio of the generated latent (`custom` uses `Custom Ratio`) — default `1:1` |
| `Custom Ratio` | string | free-form ratio like `21:9` or `21x9`, used when `Ratio` = `custom` |
| `Resolution` | combo | megapixel budget / resolution preset (`SD`, `SDXL`, `Qwen Image`, `WAN HD/FullHD`, `2–100 Megapixel`, `custom`) — default `SDXL (≈ 1024x1024)` |
| `Custom Width` / `Custom Height` | int | used when `Resolution` = `custom` (snapped to multiples of 16) |
| `Latent Channels` | combo | `4 (SD/SDXL)`, `16 (FLUX/QWEN/ZIT)`, `64 (QWEN2.1)`, `128 (FLUX2)` — default `128 (FLUX2)` |
| `Ratio From Image` | bool | use the aspect ratio of the connected `image` input instead (nearest preset ratio) |
| `Batch Size` | int | number of latents in the batch |

Additional outputs — always present, appended after the slot sockets:

- `latent` — `LATENT`, an empty latent of `batch × channels × h × w`
  (16-channel+ models use ÷16 spatial compression, smaller ones ÷8 — same
  as the standalone node)
- `width` / `height` — `INT`, the resulting pixel dimensions

The latent is generated on every run — it is cheap (zeros) and the node's
main purpose; if you don't need it, use the basic node.

## Notes

- Loading is expensive — the executor caches the node, so unchanged
  selections are not reloaded on every run; they only re-load when a widget
  value (or the preset content) changes.
- A checkpoint's own clip/VAE outputs can replace the separate encoder/VAE
  slots for classic SD-style workflows; for modern multi-model stacks
  (UNet + text encoders + VAE) leave `Checkpoint` at `(none)` and pick the
  parts individually.
- Changing a selection never touches the sockets — wires stay where they
  are. If you swap a selection, the wire simply carries the new object.
- Workflows saved with the old dynamic-socket version may have their loader
  wires on wrong slots (the old bug shifted slot indices on every save) —
  reconnect the loader outputs **once** after updating; they stay stable
  from then on.
- Presets store only widget values; if a stored filename no longer exists,
  the run fails validation with a clear "not in list" error instead of
  silently loading the wrong file.
- Presets saved before the sampler/scheduler/steps and latent fields
  existed are **migrated automatically** on load — missing keys are filled
  with their defaults and written back once.
