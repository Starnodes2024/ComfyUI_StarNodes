# ⭐ Star LoRA Extractor

## Overview
The **Star LoRA Extractor** computes the *difference* between a **base model** and a **merge model** (both from `models/diffusion_models`, the merge one having LoRAs baked in) and saves that difference as a single, normal **LoRA file** in native ComfyUI format — directly into your `models/loras` folder.

This is the classic "LoRA extraction" technique (SVD-based), the same method used by tools like kohya-ss `extract_lora_from_models`, but as a ComfyUI node:

```
diff = merged_weight − base_weight
diff ≈ lora_up @ lora_down   (per-matrix SVD at your chosen rank)
```

## Use Cases
- **Unbake LoRAs from a merge**: You downloaded a checkpoint where someone merged LoRAs in, and you want those LoRAs back as usable files.
- **Convert merges to LoRAs**: Turn an entire checkpoint merge of `base + LoRA` into a portable LoRA you can stack/swap.
- **Shrink big merges**: Extract the "flavor" of a merge as a small LoRA instead of keeping full 2–7 GB checkpoints.

## Quantized Models
The node fully supports ComfyUI-quantized model files and dequantizes each weight (including the `weight_scale` and the convrot Hadamard rotation) before diffing:
- ✅ **int8 / int8_convrot** (tensor-wise & row-wise, e.g. `*_int8_convrot.safetensors`)
- ✅ **fp8_e4m3 / fp8_e5m2** scaled files with scalar or per-row scales
- ❌ Packed 4/6-bit formats (`nvfp4`, `convrot_w4a4`, `asym_w4a8_int8`) cannot be diffed directly — use both models in bf16/fp16/fp8/int8 instead

For best quality, prefer **non-quantized (bf16/fp16) model pairs** if you have them. Note that layer diffs between two independently quantized models contain some quantization noise; the SVD truncation removes most of it.

## Very Low RAM Friendly
Both checkpoints are read **key by key** via safetensors memory-mapping — the full models are *never* loaded into RAM or VRAM at once. SVD runs one matrix at a time (in fp32). Works fine on consumer GPUs and systems with modest RAM (8–16 GB).

> **Note:** `.safetensors` checkpoints are loaded lazily (lowest RAM). Old `.ckpt`/`.pt` files must be loaded fully into RAM — convert them to safetensors first if possible.

## Inputs

### Required
| Input | Type | Description |
|---|---|---|
| **base_model** | dropdown | The clean base diffusion model, picked from `models/diffusion_models` (with fallback to `models/checkpoints` if the folder is not registered). Must be the *same architecture* as the merge model. |
| **merge_model** | dropdown | The diffusion model with LoRAs already embedded (same folder). |
| **lora_name** | text | File name for the extracted LoRA (`.safetensors` is added automatically). May contain a subfolder, e.g. `myfolder/my_lora`. |
| **dim** | int | Rank (size) of the LoRA for Linear layers. Higher = more detail + bigger file. Typical: 16–64. |
| **conv_dim** | int | Rank for Conv2d layers (UNet resnet blocks). Typically half of `dim`. |
| **precision** | dropdown | Save quality of the LoRA weights: `fp16` (default, smallest), `bf16`, or `fp32` (highest quality). |
| **device** | dropdown | Where the SVD math runs: `cpu` (default, lowest memory) or `cuda` (much faster if it fits in VRAM). |

### Optional
| Input | Type | Default | Description |
|---|---|---|---|
| **extract_convs** | bool | true | Also extract Conv2d differences (recommended for full fidelity). |
| **include_text_encoder** | bool | true | Also extract text encoder (CLIP) differences. |
| **overwrite** | bool | false | Allow overwriting an existing LoRA file with the same name. |

## Outputs
- **info** (STRING): A report with the save path, file size, and layer statistics (extracted / unchanged / skipped).

## How It Works Technically
1. Both checkpoints are opened lazily; all **matching weight keys** (2D Linear and 4D Conv2d) are iterated.
2. For each weight: `diff = merged − base` is computed in fp32.
3. Each diff matrix is decomposed with **SVD** and truncated to the chosen rank → `lora_up` / `lora_down`.
4. Keys are converted to **ComfyUI LoRA format**: diffusion-model weights become `lora_unet_…` (also for raw, unprefixed keys from single-file diffusion models, which ComfyUI wraps internally as `diffusion_model.*`), and text encoder keys (full checkpoints) become `lora_te_…`, `lora_te1_…`, `lora_te2_…` — each with `.lora_up.weight`, `.lora_down.weight` and `.alpha` entries.
5. Everything is saved as one `.safetensors` into the **top `models/loras` folder** — refresh your LoRA list and load it with the standard **Load LoRA** node.

## Size & Quality Guidelines
| Goal | dim | conv_dim | precision | Result (SDXL approx.) |
|---|---|---|---|---|
| Small & fast | 8 | 4 | fp16 | ~30–80 MB |
| Balanced | 16 | 8 | fp16 | ~60–150 MB |
| High quality | 32–64 | 16–32 | bf16/fp32 | ~150–600 MB+ |

- Extraction is an **approximation**: the merged result can also contain smoothing from re-quantization. For merges that are exactly `base + LoRA`, the recovered LoRA is usually very close to the original.
- If singular values of the diffs decay slowly (typical for merges of many strong LoRAs), use a **higher dim (32–64)** for better reconstruction at the cost of file size.
- Both models must be **the same architecture** (SD 1.5 with SD 1.5, Flux with Flux, ...). Layers with mismatched shapes are skipped automatically.

## Tips
- Run with `device: cpu` if you get out-of-memory errors; `cuda` is fine for SD 1.5/SDXL on most GPUs.
- The info output tells you how many layers were extracted vs. skipped — if almost everything was "unchanged", the two models are basically identical.
- VAE weights are always skipped (LoRAs don't target the VAE).

## Category
`⭐StarNodes/Helpers And Tools`

## Related Nodes
- **⭐ Star Dynamic LoRA Weight**: Blend/normalize multiple LoRAs on a model.
- **⭐ Star Checkpoint Saver**: Save a patched model back to a checkpoint.

---

**Unbake your merges and get portable LoRAs back! 🚀**
