# ⭐ Star Image Loader 1by1

Loads images from a folder sequentially across workflow runs, persisting state in a sidecar file.

- __Category__: `⭐StarNodes/Image And Latent`
- __Class__: `StarImageLoader1by1`
- __File__: `star_image_loader_1by1.py`

## Inputs
- __folder__ (STRING, required): Folder containing images.
- __reset_counter__ (BOOLEAN, optional, default: false): Reset sequence to the first image.
- __include_subfolders__ (BOOLEAN, optional, default: false): Traverse subfolders.
- __sort_by__ (CHOICE, optional, default: name): `name`, `date`, `size`, `random`.
- __reverse_order__ (BOOLEAN, optional, default: false): Reverse sorting order.

## Outputs
- __image__ (IMAGE): Loaded image `[1,H,W,C]` float 0..1.
- __mask__ (MASK): Simple mask (currently placeholder full-ones 64×64).
- __image_path__ (STRING): Full path of the current image.
- __current_index__ (INT): 0‑based index of the image just loaded.
- __total_images__ (INT): Total count of eligible images.
- __remaining_images__ (INT): Images left after the current one.

## Persistence
- State saved to `.star_loader_state.json` in the source folder.
- Automatically wraps to the beginning when the end is reached.

## Supported Formats
.jpg, .jpeg, .png, .gif, .webp, .bmp, .tiff, .tif

## Usage Tips
- Use `reset_counter = true` to restart from the first image.
- For shuffled order, set `sort_by = random`.

## Errors
- Folder not found
- No valid image files found

## Version
- Introduced in StarNodes 1.6+
