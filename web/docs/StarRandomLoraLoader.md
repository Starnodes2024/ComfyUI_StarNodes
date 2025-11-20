# ⭐ Star Random Lora Loader

Randomly selects and loads a LoRA from your LoRA library based on a seed value. Supports subfolder filtering, name filtering, and optional direct application to MODEL/CLIP.

This node is perfect for experimentation, batch processing with variations, or discovering LoRAs you haven't used in a while.

## Inputs

### Required

- **subfolder (CHOICE)**: Select which subfolder to search for LoRAs.
  - `All Loras` (default) – Searches all LoRAs in your entire LoRA directory, including all subfolders recursively.
  - Any subfolder name – Searches only within that specific subfolder (and its subfolders).
  - The list is automatically populated from your ComfyUI `models/loras/` directory.

- **seed (INT)**: Random seed for LoRA selection.
  - Default: `0`
  - Range: `0` to `18446744073709551615`
  - Same seed = same LoRA selection (deterministic)
  - Use `-1` or connect to a random seed generator for truly random selection each time

- **model_strength (FLOAT)**: Strength to apply the LoRA to the model.
  - Default: `1.0`
  - Range: `-10.0` to `10.0`
  - Only used if both `model` and `clip` inputs are connected

- **clip_strength (FLOAT)**: Strength to apply the LoRA to the CLIP.
  - Default: `1.0`
  - Range: `-10.0` to `10.0`
  - Only used if both `model` and `clip` inputs are connected

### Optional

- **name_contains (STRING)**: Filter LoRAs by filename.
  - Default: empty (no filtering)
  - Case-insensitive partial match
  - Example: `"anime"` will match `Anime_Style.safetensors`, `realistic_anime.ckpt`, etc.
  - Searches both filename and path

- **model (MODEL)**: Optional MODEL input to apply the LoRA directly.
  - If provided along with `clip`, the node will load and apply the selected LoRA
  - If not provided, the node only outputs the LoRA path string

- **clip (CLIP)**: Optional CLIP input to apply the LoRA directly.
  - Must be provided together with `model` for LoRA application
  - If not provided, the node only outputs the LoRA path string

## Outputs

- **model (MODEL)**: The MODEL with LoRA applied (if inputs were provided), otherwise passes through the input MODEL or None.

- **clip (CLIP)**: The CLIP with LoRA applied (if inputs were provided), otherwise passes through the input CLIP or None.

- **lora_path (STRING)**: The relative path to the selected LoRA file.
  - Format: `subfolder/filename.safetensors` or just `filename.safetensors`
  - Can be connected to standard LoRA loader nodes for more control
  - If no LoRAs match the criteria, returns an error message string

## Usage Examples

### Example 1: Random LoRA with Direct Application

Connect your checkpoint's MODEL and CLIP outputs to this node, set a seed, and the node will randomly select and apply a LoRA.

```
Checkpoint Loader → model/clip → Star Random Lora Loader → model/clip → KSampler
                                        ↓
                                   lora_path → Show Text
```

### Example 2: Random LoRA from Specific Subfolder

Set `subfolder` to `"Characters"` to only select from LoRAs in your `models/loras/Characters/` folder.

### Example 3: Filtered Random Selection

Set `name_contains` to `"style"` to only select from LoRAs with "style" in their filename.

### Example 4: Path Output Only

Don't connect MODEL/CLIP inputs. The node will output the selected LoRA path as a string, which you can:
- Display using a text display node
- Pass to a standard LoRA Loader node for manual strength control
- Use for logging or debugging

### Example 5: Batch Processing with Variations

Connect a random seed generator or use different seed values in a batch to get different LoRAs for each image in your batch.

## Behavior Details

- **File Extensions**: Searches for `.safetensors`, `.ckpt`, `.pt`, `.pth`, and `.bin` files
- **Recursive Search**: When a subfolder is selected (or "All Loras"), the search includes all nested subfolders
- **Case-Insensitive Filtering**: The `name_contains` filter is not case-sensitive
- **Deterministic Selection**: Same seed + same available LoRAs = same selection
- **Error Handling**: If no LoRAs match the criteria, the `lora_path` output will contain an error message, and MODEL/CLIP will pass through unchanged

## Tips

- Use a **Seed Generator** node set to random or increment mode for variety across multiple generations
- Combine with **name_contains** to randomly select from a themed subset (e.g., all your "anime" LoRAs)
- The string output is useful for logging which LoRA was used in each generation
- If you want more control over strength values, use the string output with a standard LoRA Loader node instead of the direct MODEL/CLIP outputs

## Notes

- This node requires ComfyUI's `folder_paths` module to locate LoRAs
- The LoRA directory is determined by ComfyUI's configuration (typically `models/loras/`)
- Subfolder list is generated at node initialization; restart ComfyUI if you add new subfolders
- LoRA application uses ComfyUI's standard `load_lora_for_models` function for compatibility
