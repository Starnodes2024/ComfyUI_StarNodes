# ⭐ Star LoRA Converter — Help

Converts an existing LoRA from `models/loras`: shrinks its rank with SVD
(Singular Value Decomposition), optionally re-casts the weights to
**FP16 / FP8 (e4m3fn)**, and saves the result as a new
`.safetensors` file — never touches the source.

The node is an `OUTPUT_NODE`, so it runs on its own — no image or latent
needed downstream. It returns a `conversion_report` string you can view
with any text-display node.

## Inputs

| Input | Options / default | What it does |
|---|---|---|
| `lora_name` | dropdown | The LoRA to convert (from `models/loras`). |
| `target_rank` | `keep`, `128`…`1` — default `32` | New rank for every LoRA module. Modules already at or below the target are kept untouched — SVD never *adds* distortion. `keep` skips rank reduction entirely (precision-only conversion). |
| `precision` | `keep`, `FP16`, `FP8 (e4m3fn)` — default `FP16` | Weight dtype of the saved file. `alpha` scalars, norms, biases and embeddings always keep a loader-safe float dtype. |
| `output_filename_prefix` | `""` | Base name of the output file. Empty = original filename. `_r<rank>` / `_fp16` / `_fp8` tags are appended automatically. |
| `save_subfolder` | `"resized"` | Subfolder inside `models/loras` the converted file is written to. |
| `svd_device` | `cuda` (default if available) / `cpu` | Where the SVD math runs — always in fp32 for numerical stability. `cpu` is the safest choice for huge text-encoder LoRAs. |

## What it does

- Detects both LoRA key conventions — `*.lora_down/lora_up` (Kohya) and
  `*.lora_A/lora_B` (Diffusers/PEFT) — plus the matching `*.alpha`.
- Handles **Linear** (2D) and **Conv2d** (4D) modules; conv tensors are
  flattened for the SVD and reshaped back afterwards.
- Merges each pair to `ΔW = up @ down` (with `alpha/rank` scaling applied),
  truncates with SVD, and splits the singular values symmetrically into the
  new `up`/`down` factors — the effective strength of the LoRA is
  preserved, only the rank changes.
- Everything that is not a LoRA pair (norms, biases, `diff`, `lokr`,
  `hada`, embeddings…) is passed through unchanged.
- Safetensors metadata is preserved; `ss_network_dim` / `ss_network_alpha`
  are updated to the new rank and a `resized_by_node` tag is added.
- Existing files are never overwritten — `_v2`, `_v3`, … is appended.

## Output

`conversion_report` lists source/output paths, the ranks found vs. the
target, resized/kept/passthrough module counts, precision, the size
comparison (MB → MB, % saved) and the processing time.
