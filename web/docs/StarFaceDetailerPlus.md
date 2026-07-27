# Star Face Detailer+

Detects faces in an image and re-renders each one at higher quality — all in a
single node. No detector loader nodes, no SAM, no extra pipes: connect **image,
model, clip, vae**, type your prompts into the widgets, and you get the finished
image back.

While it runs, the node shows a **live preview of the face currently being
detailed** plus a progress bar right inside the node body.

## Quick start

1. Put an Ultralytics **bbox** face model (e.g. `face_yolov8n.pt`) into
   `ComfyUI/models/ultralytics/bbox`.
2. *(Optional)* Put a **segm** model into `ComfyUI/models/ultralytics/segm` for
   more precise face masks.
3. Add the node (**⭐StarNodes/Sampler → ⭐ Star Face Detailer+**), wire up
   image/model/clip/vae, pick the bbox model from the dropdown, write prompts,
   queue.

## Inputs

| Input | Description |
|---|---|
| `image` | The image whose faces should be detailed. |
| `model` | Checkpoint model used for the inpainting pass. |
| `clip` | CLIP from the same checkpoint. |
| `vae` | VAE from the same checkpoint. |

## Output

| Output | Description |
|---|---|
| `image` | The final image with all selected faces refined and blended back. |

## Widgets

### Detection

| Widget | Description |
|---|---|
| `bbox_model` | Face detector from `models/ultralytics/bbox`. |
| `segm_model` | Optional segmentation model from `models/ultralytics/segm`. `none` = inpaint the plain (dilated) face box. |
| `max_faces` | Cap on how many faces are detailed — perfect for group photos. Remaining faces are left as-is. |
| `face_order` | Priority when `max_faces` limits the count: largest first, left→right, right→left, or top→bottom. |
| `bbox_threshold` | Minimum detector confidence. Lower finds more (smaller/odd) faces. |
| `bbox_dilation` | Grows/shrinks each detected box, in pixels per side. |
| `drop_size` | Faces smaller than this many pixels are skipped entirely. |
| `crop_factor` | Context around the face: crop size = face size × crop_factor. ~2.5–4 is a good range. |

### Sizing

| Widget | Description |
|---|---|
| `guide_size` | Target working resolution before detailing (ratio is always preserved). Raise it for tiny faces. |
| `guide_size_for` | `face (bbox)`: the face itself is scaled to `guide_size` — best for small faces in group shots. `crop region`: the whole crop (face + context) is scaled to `guide_size`. |
| `max_size` | Hard cap on the working resolution (VRAM/speed safety). Raise to ~1536 when detailing very small faces at high guide sizes. |

### Prompts & sampling

| Widget | Description |
|---|---|
| `positive` / `negative` | Prompts used for every face. Face-specific wording works best (e.g. “detailed beautiful face, sharp eyes”). |
| `seed` | Noise seed — the same seed is reused for every face in one run. |
| `steps`, `cfg`, `sampler_name`, `scheduler` | Standard sampling settings, applied per face. |
| `denoise` | Inpaint strength. 0.3–0.5 = gentle polish, 0.6+ = strong regeneration. |
| `feather` | Soft mask edge when blending the refined face back into the photo. |

### Per-face LoRAs

`lora_1` … `lora_5` (each with a strength slider, default `none` / 1.0).

The **first** processed face is refined with LoRA 1, the **second** with LoRA 2,
and so on. Which face is “first” depends on `face_order`. This lets you give
different characters in the same image their own LoRA in a single pass.
Faces beyond the fifth (or slots set to `none`) are refined without a LoRA.

> Tip: set `face_order` to *left to right* and line the LoRA slots up with the
> people standing in the photo.

## Tips

- **Small faces:** set `guide_size_for` to `face (bbox)` and `guide_size` to
  512–1024. The face crop is then upscaled (aspect ratio preserved, snapped to
  multiples of 8 for the VAE), refined at full detail, downscaled back with
  lanczos, and blended with a resolution-compensated feather — the surrounding
  image stays pixel-identical outside the mask.
- Keep `denoise` low (0.3–0.45) for faithful touch-ups; raise it when the source
  face is blurry or tiny.
- If faces drift from the original look, lower `denoise` before lowering `cfg`.
- The built-in ComfyUI progress bar also advances per face, so queue progress is
  visible even if the node is off-screen.
- Detector models are cached after the first run — subsequent runs start
  instantly.

## Requirements

- `ultralytics` python package (`pip install ultralytics` in your ComfyUI env).
  If it's not installed, this node is skipped and the rest of StarNodes still
  loads normally.
- An Ultralytics bbox face model in `models/ultralytics/bbox`.

Credit: inspired by FaceDetailer from
[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) by ltdrdata.
