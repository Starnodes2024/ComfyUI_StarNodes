# Star Model And AIO Merge

## Description
The **Star Model And AIO Merge** node merges up to 4 connected diffusion models into a new file — with an individual weight per model. Works with **any model architecture** (Qwen Image / Qwen Image 2.1, Flux, SD3.5, WAN, Z-Image, MiniMax H3, ...) as long as all connected models share the *same* architecture. Connect from a **UNET Loader** / **Load Diffusion Model** or from the **MODEL** output of a **Checkpoint Loader**.

Optionally you can bake up to 2 LoRAs into the merged model and plug a text encoder (CLIP connector) + VAE (VAE connector) on top, producing a complete **AIO checkpoint** — loadable directly with the standard **Load Checkpoint** node, like the files the ⭐ Star Checkpoint Saver (AIO) writes.

The merge math is done tensor by tensor **on the CPU in 32-bit precision** — no VRAM headroom needed for the merge itself. Unquantized inputs keep the dtype of the base model (`model_1`); quantized inputs are dequantized first and saved as clean BF16.

## Inputs

### Required
- **model_1** (`MODEL`)
  - Connector for the base model. Its dtype, key layout and quantization format decide what the output file looks like.
- **weight_1** (`FLOAT`, 0.0 – 1.0, default 1.0)
  - Merge weight of the base model.
- **save_name** (`STRING`, default `Merge_`)
  - File name prefix. A timestamp is appended automatically, so nothing gets overwritten: `Merge_20261110-153012.safetensors`.

### Optional
- **model_2 / model_3 / model_4** (`MODEL`)
  - Up to 3 more models to merge in. Leave unconnected to skip a slot. Connect the **raw** loader output: patches applied upstream (e.g. a LoRA Loader before this node) are **not** part of the merge — use the node's own LoRA slots below to bake LoRAs in.
- **weight_2 / weight_3 / weight_4** (`FLOAT`, 0.0 – 1.0, default 1.0)
  - Merge weight of the respective model. A weight of `0.0` disables the slot.
- **lora_1 / lora_2** (`COMBO`, default `none`)
  - LoRAs from the `loras` folder that get baked directly into the merged model weights.
- **lora_strength_1 / lora_strength_2** (`FLOAT`, -4.0 – 4.0, default 1.0)
  - Strength of the baked-in LoRA (same meaning as the model strength in a LoRA Loader).
- **text_encoder** (`CLIP`)
  - Connector for the text encoder — wire in the output of a **CLIPLoader** (or any node that outputs CLIP). Connecting it turns the output into an AIO checkpoint.
- **vae** (`VAE`)
  - Connector for the VAE — wire in the output of a **VAELoader**.

## How the weights work
The merge is a **normalized weighted average**. The weights are relative to each other — the most common combinations:

| weight_1 | weight_2 | Result |
|---|---|---|
| 1.0 | 1.0 | 50 / 50 mix |
| 1.0 | 0.5 | 67 / 33 mix |
| 0.75 | 0.25 | 75 / 25 mix |
| 1.0 | 0.0 | only model_1 (handy to just bake in LoRAs or build an AIO file!) |

Keys that exist in the base model but not in another model keep the base value.

## Where files are saved
- **Diffusion model only** → `ComfyUI/output/models/Merge_<timestamp>.safetensors`
- **With text encoder and/or VAE (AIO)** → `ComfyUI/output/checkpoints/Merge_<timestamp>.safetensors`

The merge recipe (model slots, weights, LoRAs, strengths) is stored inside the safetensors metadata as `merge_info`.

## Outputs
- **info** (`STRING`)
  - Report with the full save path, file size, how many tensors were merged and which models/LoRAs went in, plus any warnings. View it with any text preview node. The same information is printed to the console with per-tensor progress.

## Notes
- All connected models must share the same architecture. A connected model that shares **no tensors** with model #1 (e.g. accidentally plugging in a Z-Image or MiniMax model while merging Qwen) is ignored automatically and reported with a WARNING in the console and the info output — check the info text if the result looks like only one model was merged.
- **Quantized models (int8 / scaled fp8 / nvfp4 etc.) are dequantized before merging.** Mixing raw quantized codes would corrupt the weights and produce noisy/grainy images. If any input tensor was quantized (or plain FP8), that tensor is saved as clean **BF16** and stale scale/metadata keys are dropped — the output file can therefore be bigger than an int8/fp8 input. An FP8 output can be produced afterwards with the ⭐ Star FP8 Converter if needed.
- LoRA formats: standard kohya (`lora_down`/`lora_up`, with/without `alpha`) and diffusers (`lora_A`/`lora_B`) are supported, both with and without `lora_unet_`/`transformer.` prefixes. Text-encoder LoRA layers are skipped — only the diffusion model is merged.
- Restart ComfyUI or press 'r' in the model dropdowns to see the new file; it also loads directly from the output path.

## Companion nodes
- ⭐ **Star Checkpoint Saver (AIO)** — saves a loaded MODEL+CLIP+VAE pipeline as one checkpoint.
- ⭐ **Star FP16 Unet Converter** — converts a quantization to plain FP16 when you need it before merging.
- ⭐ **Star Model Packer** — merges split safetensors shards.
