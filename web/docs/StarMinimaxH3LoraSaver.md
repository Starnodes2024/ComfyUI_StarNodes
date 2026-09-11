# ⭐ Star Minimax H3 LoRA Saver — Help

Saves a merged `MINIMAX_H3_LORA` to disk in standard ComfyUI LoRA format —
the file loads with the stock **LoraLoader** (or any H3 workflow) like any
other LoRA.

- **location** — `loras folder` (default `models/loras`), `output folder`,
  or `custom path` (absolute path in `filename`).
- **filename** — default `minimax_h3_lora_merged.safetensors`.
- **overwrite** — off by default; enable to replace an existing file.

The file is written with a streaming writer (tensor-by-tensor), so even very
large merges never hold more than one tensor in RAM. The `saved_path` output
returns the absolute destination path; the loras folder file list is
refreshed after saving.

| Connector | Type | Notes |
|---|---|---|
| `minimax_h3_lora` | MINIMAX_H3_LORA | from the ⭐ Star Minimax H3 LoRA Merge node |
| `saved_path` out | STRING | absolute path of the written file |
