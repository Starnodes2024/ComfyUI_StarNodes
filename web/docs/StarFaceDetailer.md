# ⭐ Star Face Detailer

WeVisOne-VL guided refinement of faces and other regions. One VL pass detects every target and returns a precise per-instance alpha matte; each detected region is then cropped, refined at 1024px and pasted back through the soft, feathered matte. The same engine powers the per-box face detailer in ⭐ Star Regional Sampler.

- **Category**: `⭐StarNodes/Helpers And Tools`
- **Node name**: `StarFaceDetailer`
- **Outputs**: `image` (refined), `mask` (union of all pasted regions)

## Detection (`detector_model`)

WeVisOne-VL encoder from `models/text_encoders` — no ultralytics, no external node packs:

| Encoder | Size | Notes |
|---|---|---|
| `wevisone_vl_2b_int8_convrot.safetensors` | 2.7 GB | **Default**, fastest |
| `wevisone_vl_4b_int8_convrot.safetensors` | 4.9 GB | Better quality |
| `wevisone_vl_4b_bf16.safetensors` | 9.0 GB | Best quality |

A missing encoder is downloaded automatically from [Comfy-Org/WeVisOne-VL](https://huggingface.co/Comfy-Org/WeVisOne-VL) on first use. Workflows saved with older YOLO detector selections fall back to the default encoder with a console note.

## Refine targets

`refine_target`: `face` (default), `eyes`, `mouth`, `nose`, `ears`, `breast`, `body`, or `custom` with a free-text `custom_target` (e.g. `hands`, `lips`, `tattoo`). Every detected instance is refined one by one.

## Main settings

- **prompt / negative_prompt** — default: `high quality face details, natural skin texture, fine pores, photorealistic, preserve original skin tone and facial identity`
- **steps** 12, **cfg** 1.0, **euler / simple**, **denoise** 0.5 (0.3–0.4 keeps more identity)
- **guide_size** 1024 — every crop is resized to this square before refinement (0 = auto)
- **crop_factor** 1.2 — tight square around the detection
- **feather** 16 — blend width, auto-scales with crop size
- **color_match** 1.0 — per-channel mean/std color transfer before pasting, keeping the original skin tone/lighting. With `reference_image` it uses whole-crop statistics (harmonizes a swap without recoloring the new identity); 0 = off

## Optional inputs

- **positive / negative** (`CONDITIONING`) — reuse the generation's conditioning (overrides the prompt text; a console note tells you when this happens)
- **prompt_refiner** — any text encoder: VL models (Qwen3-VL, Qwen3.5, WeVisOne, Gemma) look at each crop and write a short per-face prompt combined with your prompt text; non-VL models refine the prompt text only. Printed to the console per face
- **face_prompts** (`STAR_FACE_PROMPTS`) — from ⭐ Star Multi Face Details: per-face custom prompts, per-face LoRAs with strength, and per-face skipping
- **reference_image** — identity swap for Qwen Image 2.1 / Fibo2 / MOSS Image Edit (denoise 0.75–1.0; color match switches to whole-crop mode)
- **preview** (`STAR_PREVIEW`) — live sampling preview via ⭐ Star Preview

## ⭐ Star Multi Face Details

Companion node with 5 slots (detection order, left to right): `prompt_1–5`, `lora_1–5` + `strength_1–5`, `lora_search` (live dropdown filter), and `skip_faces` (e.g. `1, 3` to leave faces unrefined). Slot prompts also drive the prompt refiner individually. See `StarMultiFaceDetails.md`.
