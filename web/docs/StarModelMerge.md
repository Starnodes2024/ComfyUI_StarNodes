# ⭐ Star Model Merge

Merges 2 to 4 diffusion models with individual weights into a new `.safetensors` file and saves it into a subfolder of the ComfyUI output folder. Works with any diffusion model loadable in ComfyUI (e.g. moss-image, Flux, SDXL, ...). The merged file can be loaded again with the standard **Load Diffusion Model** node.

- **Category**: `⭐StarNodes/Helpers And Tools`
- **Node name**: `StarModelMerge`
- **Output**: `info` (STRING)

## Inputs

- **model_1** / **model_2** (`MODEL`)
  - The two base models to merge (required). Connect e.g. from a **Load Diffusion Model** node.

- **model_3** / **model_4** (`MODEL`, optional)
  - Additional models to include in the merge.

- **weight_1** ... **weight_4** (`FLOAT`, default `1.0`, range `0.0` – `10.0`, step `0.01`)
  - Per-model weight. Weights are normalized per tensor: `merged = sum(weight_i * tensor_i) / sum(weights)`, so only the ratio matters. Weights are per-tensor re-normalized if a key is missing in one of the models. A weight of `0` skips that model completely.

- **output_dtype** (dropdown, default `keep first model`)
  - `keep first model`: saved file keeps the dtype of model_1 (recommended, e.g. BF16 stays BF16).
  - `bf16` / `fp16` / `fp32`: force a specific dtype.

- **save_folder** (`STRING`, default `merges`)
  - Subfolder inside the ComfyUI output folder where the merged file is saved. Created automatically.

- **save_name** (`STRING`, default `merged_model`)
  - File name; `.safetensors` is appended if missing. Existing files are never overwritten — a counter (`_0002`, `_0003`, ...) is appended instead.

## Behavior

1. Reads the raw diffusion-model weights of every connected model (the `diffusion_model.` prefix is stripped so the file loads back cleanly).
2. Merges tensor by tensor on CPU in float32 for precision, then casts back to the chosen output dtype.
3. Reports keys that are missing in some models (their weights are re-normalized) and shape mismatches (skipped with a warning in the info output).
4. Saves the merged state dict as safetensors and returns an **info** string with all details: models, architectures (warns if they differ), weights, tensor counts, output path, file size and duration.

Models must have their weights actually loaded (standard loading via *Load Diffusion Model* is enough). Merging is deterministic — the same inputs always produce the same file.
