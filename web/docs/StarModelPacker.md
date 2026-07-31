# ⭐ Star Model Packer

## Overview
The Star Model Packer node combines split safetensors model files into a single file and converts them to the specified floating-point precision (FP8, FP16, or FP32).

## Use Case
This node is useful when you have:
- Split model files (e.g., `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, etc.)
- Models downloaded from Hugging Face that are split into multiple parts
- Models that need to be converted to a different precision for memory optimization or compatibility

## Inputs

### Required
- **input_folder** (STRING)
  - Full path to the folder containing split safetensors files
  - Example: `D:/AI/ComfyUI202511/HF/qwen3-4b-heretic-zimage/qwen-4b-zimage-heretic`
  - All `.safetensors` files in this folder will be combined

- **precision** (DROPDOWN)
  - Target floating-point precision for the output model
  - Options:
    - **FP8** (default) - 8-bit floating point (e4m3fn format) - Smallest size, ~50% reduction
    - **FP16** - 16-bit floating point - Balanced size/quality
    - **FP32** - 32-bit floating point - Full precision, largest size
  
- **save_name** (STRING, optional)
  - Custom name for the output file
  - If left empty, automatically generates name as: `{folder_name}_{precision}.safetensors`
  - Example: If folder is `qwen-4b-zimage-heretic` and precision is FP8, output will be `qwen-4b-zimage-heretic_FP8.safetensors`
  - `.safetensors` extension is added automatically if not provided

## Outputs

- **status** (STRING)
  - Detailed status message containing:
    - Output filename
    - Full output path
    - Number of input files processed
    - Total input size
    - Output size
    - Selected precision
    - Total number of tensors

## Features

### Automatic File Discovery
- Automatically finds all `.safetensors` files in the input folder
- Sorts files alphabetically for consistent processing
- Reports the number of files found

### Progress Tracking
- Displays detailed progress in the console:
  - Lists all files being processed
  - Shows loading progress for each file
  - Reports conversion progress
  - Displays summary statistics

### Smart Tensor Handling
- Combines all tensors from multiple files
- Detects and warns about duplicate tensor keys
- Converts floating-point tensors to target precision
- Preserves non-floating-point tensors (e.g., integer indices)
- Handles conversion errors gracefully

### Memory Efficiency
- Processes tensors in batches
- Shows progress every 100 tensors during conversion
- Cleans up intermediate data

### File Size Reporting
- Calculates total input size across all files
- Reports output file size
- Displays sizes in both MB and GB for large models

## Output Location

Models are saved to: `{ComfyUI}/output/models/`

## Example Usage

### Basic Usage (Auto-naming)
```
Input Folder: D:/AI/ComfyUI202511/HF/qwen3-4b-heretic-zimage/qwen-4b-zimage-heretic
Precision: FP8
Save Name: (empty)

Output: {ComfyUI}/output/models/qwen-4b-zimage-heretic_FP8.safetensors
```

### Custom Naming
```
Input Folder: D:/AI/ComfyUI202511/HF/my-model/split-files
Precision: FP16
Save Name: my_custom_model_fp16

Output: {ComfyUI}/output/models/my_custom_model_fp16.safetensors
```

## Console Output Example

```
============================================================
Star Model Packer - Starting
============================================================
Found 3 safetensors file(s)
  1. model-00001-of-00003.safetensors
  2. model-00002-of-00003.safetensors
  3. model-00003-of-00003.safetensors

Target precision: FP8
Output file: D:/ComfyUI/output/models/qwen-4b-zimage-heretic_FP8.safetensors

============================================================
Loading and combining tensors...
============================================================

Loading file 1/3: model-00001-of-00003.safetensors
  Found 150 tensor(s)
  Progress: 150 total tensors loaded

Loading file 2/3: model-00002-of-00003.safetensors
  Found 150 tensor(s)
  Progress: 300 total tensors loaded

Loading file 3/3: model-00003-of-00003.safetensors
  Found 100 tensor(s)
  Progress: 400 total tensors loaded

============================================================
Converting to FP8...
============================================================
  Converting tensor 100/400...
  Converting tensor 200/400...
  Converting tensor 300/400...
  Converting tensor 400/400...

Conversion complete:
  Converted: 380 tensors
  Skipped: 20 tensors (non-floating or failed)

============================================================
Saving packed model...
============================================================
Successfully saved to: D:/ComfyUI/output/models/qwen-4b-zimage-heretic_FP8.safetensors

============================================================
Summary
============================================================
Input files: 3
Total input size: 8.50 GB (8704.00 MB)
Output size: 4.25 GB (4352.00 MB)
Precision: FP8
Total tensors: 400
============================================================
```

## Tips

1. **Precision Selection**:
   - Use **FP8** for maximum memory savings (recommended for large models)
   - Use **FP16** for a balance between size and quality
   - Use **FP32** only if you need full precision

2. **Folder Organization**:
   - Keep all split files in a dedicated folder
   - Ensure all parts of the model are present

3. **Naming Convention**:
   - Leave `save_name` empty for automatic naming based on folder name
   - Use custom names when you want specific output filenames

4. **Error Handling**:
   - Check console output for detailed progress and any warnings
   - The node will report if files are missing or conversion fails

## Category
⭐StarNodes/Helpers And Tools
