# ⭐ Star Save Folder String

Build a portable path string for saving images with optional date-based folders and timestamped filenames.

This node outputs a single `STRING` representing a path using forward slashes (portable across OS). You can send this into any node that accepts a string path or use it as a base for save nodes.

## Inputs

- **preset_folder (CHOICE)**: First-level folder. `None` or one of your presets loaded from JSON.
- **date_folder (BOOLEAN)**: If true, prepends a date folder `YYYY-MM-DD/` after the preset.
- **date_folder_position (CHOICE)**: Where to place the date folder when enabled.
  - `first` – date is the first segment (default): `YYYY-MM-DD/...`
  - `subfolder` – date is placed after preset/custom folders: `Preset/Custom/.../YYYY-MM-DD`
- **custom_folder (STRING)**: Top-level folder. Used only if not empty. Default `ComfyUi`.
- **custom_subfolder (STRING)**: Optional subfolder inside the custom folder.
- **date_in_filename (CHOICE)**: Add date to filename.
  - `Off` – no date in filename
  - `prefix` – `YYYY-MM-DD` before the filename
  - `suffix` – `YYYY-MM-DD` after the filename
- **filename (STRING)**: Base filename. Uses `Image` if empty.
- **add_timestamp (BOOLEAN)**: If true, append `_HH-MM-SS` after the filename/date. Default off.
- **separator (STRING)**: Separator between filename and timestamp when `add_timestamp` is true. Default `_`.

## Outputs

- **path (STRING)**: The constructed path string (always using `/` as the separator).

## Behavior details

- Invalid filename characters (`<>:"/\\|?*` and whitespace control) are removed from inputs.
- If all folder inputs are empty and `date_folder` is false, the output will be just the filename portion.
- Time format is `HH-MM-SS`, date `YYYY-MM-DD`.

## Examples

1. date_folder = true, custom_folder = `folder`, custom_subfolder = `subfolder`, date_in_filename = `suffix`, filename = `Image`, add_timestamp = true, separator = `_` →
   `2025-08-20/folder/subfolder/Image2025-08-20_18-23-59`

2. date_folder = false, preset_folder = `Images`, custom_folder empty, date_in_filename = `prefix`, filename = `Cat`, add_timestamp=false →
   `Images/2025-08-20Cat`

3. preset_folder = `Images`, date_folder = true, date_folder_position = `subfolder`, custom_folder = `Renders`, custom_subfolder = `A`, filename = `Pic`, Off →
   `Images/Renders/A/2025-08-20/Pic`

## Notes

- This node only builds a path string; it does not create folders or write files.
- Use with your preferred save node; most will create missing folders automatically or expose a base directory option.

## Preset folders

- The first path segment can be chosen from a presets list using the `preset_folder` input. `None` means no preset is used.
- Presets are loaded from a JSON file at:
  `custom_nodes/ComfyUI_StarBetaNodes/star_save_folder_presets.json`
- Default presets included: `Images, Video, Inpaint, Controlnet, Edits, Faceswap, Creatures, Backdrops`.
- You can edit this file to add or remove strings. It must contain a JSON array of unique strings, for example:

```json
[
  "Images",
  "Projects",
  "Clients"
]
```

- Save the file and refresh ComfyUI (or reload custom nodes) for changes to appear in the dropdown.
