# Star FP8 Converter

## Description
The **Star FP8 Converter** node converts an existing `.safetensors` checkpoint to FP8 (`float8_e4m3fn`) and writes the converted file into the standard ComfyUI output directory under a `models` subfolder.

Use this when you want to experiment with FP8-quantized weights for models that ComfyUI already supports in FP8 format.

## Inputs

### Required
- **model_path** (`STRING`)
  - Full path to the source `.safetensors` file you want to convert.
  - Example: `F:/ComfyUIModels/models/clip/qwen_3_4b.safetensors`.
- **save_name** (`STRING`)
  - Base name for the converted file. The node appends `.safetensors` if missing.
  - Example: `qwen_3_4b_fp8_e4m3fn`.

## Outputs
- **status** (`STRING`)
  - A short message indicating success or describing any error that occurred.
  - On success, it includes the relative path of the converted FP8 model inside the ComfyUI output directory, e.g. `models/qwen_3_4b_fp8_e4m3fn.safetensors`.

## Behavior
1. Validates the provided `model_path` and `save_name`.
2. Loads the source `.safetensors` file.
3. Converts all floating-point tensors to `torch.float8_e4m3fn`, keeping non-floating tensors unchanged.
4. Saves the converted checkpoint to:
   - `<ComfyUI output>/models/<save_name>.safetensors`.
5. Returns a status string you can inspect with a text preview node.

## Notes
- You should only use this node for models that ComfyUI (and your GPU / PyTorch build) can actually run in FP8.
- FP8 is much more aggressive than FP16, so image or text quality may change compared to the original checkpoint. Always keep a backup of the original file.
