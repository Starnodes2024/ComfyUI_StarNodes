# ⭐ Star Multi Face Details

Companion node for the ⭐ Star Face Detailer: write custom prompts for up to 5 faces/objects individually instead of using one general prompt for every detected region. Each prompt can also drive the detailer's prompt refiner one by one.

- **Category**: `⭐StarNodes/Helpers And Tools`
- **Node name**: `StarMultiFaceDetails`
- **Output**: `face_prompts` (`STAR_FACE_PROMPTS`)

## Inputs

- **lora_search** (`STRING`, optional)
  - Type to filter all five LoRA dropdowns live (case-insensitive substring). Clear to restore the full list. The current selection stays visible even when it does not match the filter.

- **prompt_1 … prompt_5** (`STRING`, multiline, all optional)
  - Custom prompt for face/object 1–5 in detection order (left to right, top to bottom).
  - Empty slots fall back to the detailer's general prompt.

- **lora_1 … lora_5** (`COMBO`, default `none`)
  - Optional LoRA applied to the diffusion model **only for that face's refine pass** — the base model and all other faces are unaffected. Works with or without a custom prompt in the same slot, and together with the prompt refiner. Weights are loaded from disk once and cached.

- **strength_1 … strength_5** (`FLOAT`, default `1.0`, range `-10.0` – `10.0`)
  - Strength of the slot's LoRA. `0` disables it.

- **skip_faces** (`STRING`, optional)
  - Face/object numbers to leave completely unrefined, e.g. `1, 3` (detection order).
  - Skipped faces are not sampled at all and are excluded from the detailer's `mask` output — ideal to refine only one face in a group shot (skip all others).

## Usage

Connect `face_prompts` to the ⭐ Star Face Detail node's optional `face_prompts` input.

For each detected face/object (see the detailer's `refine_target`):

- **Without prompt refiner**: a non-empty slot prompt is encoded directly and used as the positive conditioning for that face, overriding the general prompt and the connected `positive` input.
- **With prompt refiner enabled**: the slot prompt becomes the user instructions for the VL refinement of that face — the VL sees the face crop and writes a short, tailored prompt guided by *that slot's* text.
- Every used prompt (custom or refined) is printed to the console.

Works for all refine targets (`face`, `eyes`, `mouth`, ..., `custom`) — the slots always follow detection order, so slot 1 belongs to the first detected region, slot 2 to the second, and so on.
