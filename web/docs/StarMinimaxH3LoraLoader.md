# ⭐ Star Minimax H3 LoRA Loader — Help

Loads a MiniMax-H3 LoRA `.safetensors` from `models/loras` into a
`MINIMAX_H3_LORA` object for the merge pipeline.

Handles `lora_up`/`lora_down`/`alpha` as well as the `lora_A`/`lora_B`
aliases and reports incomplete pairs in the `report` output.

```
[LoRA Loader A] ─┐
                  ├─> [⭐ Star Minimax H3 LoRA Merge] ─> [⭐ Star Minimax H3 LoRA Saver]
[LoRA Loader B] ─┘
```

## Outputs

| Output | Type | Notes |
|---|---|---|
| `minimax_h3_lora` | MINIMAX_H3_LORA | the loaded LoRA state dict |
| `report` | STRING | pair count / extra tensors / dangling pairs |
