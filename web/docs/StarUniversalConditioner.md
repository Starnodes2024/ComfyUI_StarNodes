# ⭐ Star Universal Conditioner

One conditioning node for **every image and image-edit model**: encodes the
positive and negative prompt and feeds connected reference images the way each
model family expects — for **text-to-image** and **edit** workflows alike.

Works with MOSS-Image, Fibo2, Flux Klein, Flux 2 / Flux 3, Krea 2, Qwen Image
(Edit / 2.1), SDXL and friends — the same node, no per-model switching.

## Inputs

| Input | Type | Purpose |
|---|---|---|
| `clip` | CLIP | required — the model's text encoder (from the Star Model Loader or any CLIP source) |
| `vae` | VAE | optional — needed to turn reference images into the `reference_latents` edit models consume. Without it, images only reach vision-language text encoders |
| `Image 1…10` | IMAGE | optional reference images — ten fixed sockets, used in order (fixed sockets keep workflow links stable) |

## Widgets

| Widget | Type | What it does |
|---|---|---|
| `positive` | string | the positive prompt |
| `negative` | string | the negative prompt — left empty it produces each model's own empty/null conditioning (e.g. Fibo2's null caption) |
| `Negative Zero` | bool | zeroes the negative conditioning exactly like the native `ConditioningZeroOut` node (zeroed cross-attention + pooled fields — reference latents are kept) |
| `Reference Resize` | combo | reference images are scaled to about this many megapixels (aspect preserved, 32 px grid) before VAE encode / the vision tower - important for image-edit models. `none` keeps each image at its own resolution. Options: `none`, `0.5 MP`, `1 MP` (default), `1.5 MP`, `2 MP`, `3 MP` |

## Outputs

- `positive` / `negative` — `CONDITIONING` for the sampler

## How it adapts per model

- **Vision-language encoders** (Qwen Image, Qwen Image 2.1, Krea 2,
  MOSS-Image, Fibo2, Flux 3, Boogu, JoyImage, MageFlow, LTXV): the images are
  passed into `tokenize()`, so the vision tower sees them alongside the text.
  Detected automatically from the tokenizer — no setting needed.
- **Reference latents**: with a VAE connected, every image is encoded and
  appended to **both** conditionings as `reference_latents` — the standard key
  used by Flux/Flux2/Flux3, Qwen Image, Krea2, MOSSImage, Fibo2 and the other
  edit models (same behaviour as the native Set Reference Multi Latent node).
- **Non-edit encoders** (SDXL, Flux 2 dev/Klein text path): images are skipped
  for tokenize; `reference_latents` are still appended for models that read
  them — models that don't simply ignore the key.
- **Qwen Image 2.1** specifics are handled too: alpha is composited over white
  for the vision tower while the VAE keeps all channels, and vision tokens stay
  in the text embed only when no reference latents exist (`keep_vision`).

## Notes

- Text-to-image: connect only `clip` (and write prompts) — the node then
  behaves like a normal dual text encode.
- Edit: connect `clip` + `vae` + the image(s) — everything is wired for the
  model automatically.
- Up to 10 reference images per conditioner; chain another conditioner for
  more (the later one's `reference_latents` append).
