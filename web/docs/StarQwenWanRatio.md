# ⭐ Star Qwen / WAN Ratio

Generate a latent tensor and width/height based on the selected model and aspect ratio.
This node extends the original `⭐ Star Qwen Image Ratio` by adding support for WAN HD and WAN Full HD targets, while preserving Qwen's original ratio presets.

- Qwen uses the exact dimensions from `⭐ Star Qwen Image Ratio`.
- WAN HD aims for a total of 1280×720 (921,600) pixels.
- WAN Full HD aims for a total of 1920×1080 (2,073,600) pixels.
- All generated sizes are multiples of 16. Latent grid sizes are multiples of 8.

## Inputs

- __model__
  - Options: `Qwen`, `Wan HD`, `Wan Full HD`
  - Select which preset list to use.

- __qwen_ratio__ (used when model = `Qwen`)
  - Options include labels like `1:1 (1328x1328)`, `16:9 (1664x928)`, etc., matching the original Qwen node.
  - Includes `Free Ratio (custom)` to use custom width/height.

- __wan_hd_ratio__ (used when model = `Wan HD`)
  - Aspect options are the same as Qwen, but each label displays the computed width×height that best fits a total of 921,600 pixels.
  - Includes `Free Ratio (custom)`.

- __wan_fhd_ratio__ (used when model = `Wan Full HD`)
  - Aspect options are the same as Qwen, but each label displays the computed width×height that best fits a total of 2,073,600 pixels.
  - Includes `Free Ratio (custom)`.

- __batch_size__
  - Integer. Number of latent samples to output.

- __custom_width__, __custom_height__
  - Used when a `Free Ratio (custom)` option is selected.
  - Values are clamped to multiples of 16 internally.

- __use_nearest_image_ratio__
  - Boolean. When enabled and an image is connected, the node ignores the dropdown ratio and automatically selects the preset whose aspect ratio is closest to the input image for the chosen `model`.
  - For `Wan HD` and `Wan Full HD`, this means you’ll get sizes like `1280x720` or `1920x1080` when the image is ~16:9, `720x1280` or `1080x1920` for ~9:16, etc.

- __image__ (optional)
  - `IMAGE`. If provided and `use_nearest_image_ratio` is ON, the image’s aspect ratio is used to choose the nearest preset.
  - If no image is connected or the toggle is OFF, the node uses the selected dropdown.

## Outputs

- __latent__ (LATENT)
  - Zero-initialized latent with shape `[batch, 4, H/8, W/8]`.
- __width__ (INT)
- __height__ (INT)

## Notes

- For `Wan HD` and `Wan Full HD`:
  - The node computes sizes in multiples of 16 that closely match the desired total pixel count while respecting the chosen aspect ratio.
  - Common targets (e.g., `16:9` and `9:16`) prefer exact HD/FHD sizes like `1280x720`, `1920x1080`, `720x1280`, and `1080x1920`.
- The latent grid uses an 8× downsample factor, so the tensor spatial size is `width/8` × `height/8`.

- When `use_nearest_image_ratio` is enabled and an image is connected, nearest selection by aspect ratio takes precedence over dropdown choices. If anything goes wrong (e.g., missing image), the node falls back to the dropdown or custom size.

## Example

- Set `model = Wan HD` and `wan_hd_ratio = 16:9 (1280x720)` to get HD landscape.
- Set `model = Wan Full HD` and `wan_fhd_ratio = 9:16 (1080x1920)` to get Full HD portrait.
- Set `model = Qwen` and pick any of the original Qwen ratios to mirror the prior behavior.
- Connect an image, enable `use_nearest_image_ratio`, and set `model = Wan HD` to auto-pick a size that best matches the image aspect (e.g., ~16:9 → `1280x720`).
