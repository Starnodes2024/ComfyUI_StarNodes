# Star FP16 Unet Converter

## Description
The **Star FP16 Unet Converter** node takes a model from your `diffusion_models` folder (the same list the standard **Load Diffusion Model / UNET Loader** node shows, e.g. `H:\4Comfyui\models\diffusion_models`), dequantizes its weights and saves a plain FP16 copy next to the original file.

The converted file is saved **in the same folder**, with `_FP16` added to the filename:
- `flux1-dev-fp8-e4m3fn.safetensors` → `flux1-dev-fp8-e4m3fn_FP16.safetensors`

Use this when you need a real FP16 version of a quantized model — for example for merging, LoRA training pipelines, TensorRT conversion, or older tools that cannot read FP8 checkpoints.

## Inputs

### Required
- **model_name** (`COMBO`)
  - One of the models in your `diffusion_models` folder(s). This is the same dropdown list as the UNET Loader, so subfolders are supported too.
- **device** (`CPU` / `GPU`)
  - `CPU` (default): the weights are converted in system RAM. Recommended for large models — a FP16 copy of a big UNet needs roughly double the source file size in RAM (e.g. ~24 GB RAM for a 12 GB FP8 model).
  - `GPU`: the dequantize/cast math runs on the graphics card for extra speed. Only the currently processed tensor is moved to the GPU, so VRAM usage stays low.

## Outputs
- **status** (`STRING`)
  - A report with the output path, old/new file sizes, how many quantized layers were dequantized and how many tensors were converted. View it with any text preview node.

## Supported formats
The node accepts every checkpoint container the UNET Loader can open (`.safetensors`, `.sft`, `.ckpt`, `.pt`, `.pt2`, `.pth`, `.bin`) and understands the FP8 quantization variants ComfyUI can load for diffusion models:

- **Plain FP8** checkpoints (`float8_e4m3fn` / `e5m2` weights without scales, e.g. files from the ⭐ Star FP8 Converter)
- **Scaled FP8, old format** (`scaled_fp8` marker + `*.scale_weight` entries, e.g. older Flux/WAN fp8 checkpoints)
- **Scaled FP8, new ComfyUI format** (`_quantization_metadata` in the safetensors header or per-layer `.comfy_quant` entries, with `*.weight_scale` scalars)
- **INT8 tensor-wise** quantized weights (`int8_tensorwise` with `*.weight_scale`)
- Unquantized FP32/BF16 files are simply cast down to FP16.

Quantization helper entries (`weight_scale`, `input_scale`, `comfy_quant`, `scaled_fp8`, …) are removed from the saved file and the `_quantization_metadata` header entry is dropped, so the result is a clean plain-FP16 checkpoint that loads everywhere.

Block-/group-quantized formats like **NVFP4**, **MXFP8** or 4-bit INT (w4a8) are detected but intentionally not converted — dequantizing them back to FP16 is lossy and bloated compared to the original; the node will stop with a clear message naming the affected layer instead of writing a broken file.

## Notes
- The original file is never touched. If a `<name>_FP16.safetensors` file already exists, the node refuses to overwrite it — rename or delete the old file first.
- After converting, restart ComfyUI or refresh the page so the new file shows up in the model dropdowns.
- FP8 → FP16 cannot bring back the information lost during FP8 quantization; the result is what a dequantized FP8 model would compute, just stored in a universally loadable format.
- GGUF checkpoints are not supported (they need the separate GGUF loader nodes and cannot be written back as GGUF by this node).

## Companion nodes
- ⭐ **Star FP8 Converter** — the opposite direction: converts a `.safetensors` checkpoint to FP8.
- ⭐ **Star Model Packer** — merges split safetensors shards and converts precision at the same time.
