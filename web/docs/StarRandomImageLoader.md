# ⭐ Star Random Image Loader

Loads a random image (and mask if present) from a folder. Supports optional subfolder traversal and deterministic selection via seed.

- __Category__: `⭐StarNodes/Image And Latent`
- __Class__: `StarRandomImageLoader`
- __File__: `star_random_image_loader.py`

## Inputs
- __folder__ (STRING, required): Path to the folder containing images.
- __include_subfolders__ (BOOLEAN, optional, default: false): Search recursively.
- __seed__ (INT, optional, default: 0): 0 = random each run; non‑zero = deterministic selection.

## Outputs
- __image__ (IMAGE): Loaded RGB image tensor `[1,H,W,3]` float 0..1.
- __mask__ (MASK): If source has alpha channel, mask is derived from A; else zeros.
- __image_path__ (STRING): Full path of the selected image.
- __seed__ (INT): The seed used for selection (UI extra output is displayed as "seed").

## Supported Formats
.jpg, .jpeg, .png, .gif, .webp

## Behavior
- Files are sorted to ensure consistent ordering.
- If `seed == 0`, a random seed is generated per run; otherwise selection is deterministic.
- Alpha channel, when present, is converted to a mask (inverted to match ComfyUI convention).

## Usage Tips
- __Deterministic pick__: Set a specific `seed` to always load the same image for a given folder state.
- __Shuffle__: Use `seed = 0` for a different random image each execution.
- __Recursive__: Enable `include_subfolders` to expand the image pool.

## Errors
- Folder not found
- No valid images in directory

## Version
- Introduced in StarNodes 1.6.0
