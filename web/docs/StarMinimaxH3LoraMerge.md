# ⭐ Star Minimax H3 LoRA Merge — Help

Merges two `MINIMAX_H3_LORA` objects into one normal LoRA that loads with the
stock **LoraLoader** like any other LoRA.

- **weight** — blend slider: `0` = only LoRA A, `0.5` = 50/50, `1` = only
  LoRA B.
- **output_rank** — rank of the merged LoRA (dropdown 8–128, default 32). If
  the combined rank of A+B is smaller, that is used instead — nothing is
  lost.
- **output_dtype** — storage dtype: `bf16` (default) / `fp16` / `fp32`.

How it works: both LoRAs' deltas are computed in fp32 (alpha/r scaling
included), blended in weight space, then re-composed into up/down factors via
(low-rank) SVD — so the result stays a standard LoRA. Keys present in only
one LoRA are carried over, scaled by their side's weight. Merge math is
always fp32; `output_dtype` only controls storage.

LoRA keys are matched by exact base-layer names, so two H3 LoRAs only ever
blend on layers they actually share.

## Connectors & outputs

| Connector | Type | Notes |
|---|---|---|
| `lora_a` / `lora_b` | MINIMAX_H3_LORA | from two ⭐ Star Minimax H3 LoRA Loader nodes |
| `minimax_h3_lora` out | MINIMAX_H3_LORA | merged LoRA — feed the Saver |
| `report` out | STRING | merge statistics |
