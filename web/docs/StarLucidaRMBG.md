# ⭐ Star Lucida RMBG — Help

Background removal node powered by **Lucida** — a BiRefNet fine-tune that excels at cases where general removers fail: **semi-transparent objects (glass), camouflage, text & logos with soft shadows, glow/VFX effects, and illustrations / line art**.

**Credit:** Based on [Lucida](https://github.com/egeorcun/lucida) by [egeorcun](https://github.com/egeorcun). Model weights from [huggingface.co/egeorcun/lucida](https://huggingface.co/egeorcun/lucida) (MIT license).

---

## Features

- **Automatic model download** — The model (`model.safetensors`, ~885 MB) downloads automatically from HuggingFace on first use and is saved to `ComfyUI/models/lucida/`.
- **Single-model cache** — The model stays loaded between runs for fast repeated executions.
- **Batch processing** — Processes image batches one by one.
- **RGBA output** — Returns both a 4-channel RGBA cut-out (save as PNG to keep transparency) and a separate alpha mask for compositing.

---

## Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| **image** | IMAGE | — | Input image (batches are processed one by one) |
| **refine** | BOOLEAN | `false` | Edge-refinement pass: re-runs the model on the uncertain alpha band at higher effective resolution. Slower, but produces sharper edges. Matches the upstream `--refine` option. |
| **decontaminate** | BOOLEAN | `true` | Removes background colour spill from the cut-out RGB channels. Uses `pymatting` if installed, otherwise falls back to a built-in nearest-colour method. Matches the upstream default. |
| **dtype** | DROPDOWN | `fp32` | Precision: `fp32` matches the reference pipeline; `bf16`/`fp16` roughly halve VRAM usage on GPU. CPU always runs `fp32`. |
| **device** | DROPDOWN | `auto` | Device selection: `auto` uses ComfyUI's device, or force `cuda` / `cpu`. |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| **rgba_image** | IMAGE | 4-channel RGBA cut-out. Save with **Save Image** as PNG to preserve transparency, or feed to compositing nodes. |
| **mask** | MASK | Raw soft alpha matte for use with `ImageCompositeMasked`, inpainting, or other masking operations. |

---

## Technical Details

- **Inference resolution:** The model runs at its trained resolution of 1024×1024 with ImageNet normalization (identical to the official repo). The alpha is resized back to your image's original resolution.
- **GPU memory:** Roughly 4–6 GB in `fp32` at 1024×1024. Use `bf16` or `fp16` if you're tight on VRAM.
- **Architecture:** The bundled code (`lucida/birefnet.py`, `lucida/BiRefNet_config.py`) is the original from the Lucida/BiRefNet HF repos (MIT), with only `timm`/`kornia` imports made optional. Weights load with `strict=True`.

---

## Dependencies

The following dependencies are required (automatically installed with StarNodes):

- `transformers>=4.38.0`
- `safetensors>=0.4.0`
- `einops>=0.6.0`

Optional (for exact official color-decontamination quality):

- `pymatting` — Without it, the node uses a built-in nearest-colour fallback and prints a note to the console.

---

## Attribution

- **Architecture and base weights:** [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT)
- **Fine-tune:** [egeorcun/lucida](https://github.com/egeorcun/lucida) (MIT)
- **Original ComfyUI node:** [ComfyUI-Lucida](https://github.com/egeorcun/lucida)

---

## License Notes

- The model and code are MIT licensed.
- Some of Lucida's training datasets are research-only. See the [upstream README](https://github.com/egeorcun/lucida) for details if you plan commercial use.
