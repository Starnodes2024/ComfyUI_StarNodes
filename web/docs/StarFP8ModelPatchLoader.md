# ⭐ Star FP8 Model Patch Loader

## Overview

`StarFP8ModelPatchLoader` is a StarNodes replacement for ComfyUI's `ModelPatchLoader` that is safe to use with FP8 (`float8_e4m3fn` / `float8_e5m2`) tensors.

It works around a PyTorch limitation where `torch.count_nonzero()` is not implemented for CPU FP8 tensors. Some model patch files trigger this path during loading.

## Category

`⭐StarNodes/Loaders`

## Inputs

- **name**
  - **Type:** dropdown (`model_patches`)
  - **Description:** The patch file to load from your ComfyUI `model_patches` folder.

## Outputs

- **MODEL_PATCH**
  - A ComfyUI `ModelPatcher` containing the loaded patch model.

## What problem it solves

When loading certain model patches in ComfyUI, the original loader can fail with an error similar to:

`"nonzero_count_cpu" not implemented for 'Float8_e4m3fn'`

This node avoids the crash by performing the internal "is this tensor all zeros?" check using a safe dtype conversion when the tensor is FP8 on CPU.

## Notes

- This node is intended as a drop-in replacement for `ModelPatchLoader`.
- You must restart ComfyUI after updating StarNodes in order for the new node and documentation to appear.
