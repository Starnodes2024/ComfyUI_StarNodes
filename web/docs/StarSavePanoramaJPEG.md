# ⭐ Star Save Panorama JPEG

Saves images as JPEG with embedded XMP panorama metadata for panorama viewers.

- __Category__: `⭐StarNodes/Image And Latent`
- __Class__: `StarSavePanoramaJPEG`
- __File__: `star_save_panorama_jpeg.py`

## Inputs
- __images__ (IMAGE, required): Batch of images to save.
- __filename_prefix__ (STRING, required, default: "ComfyUI"): Prefix for saved files.
- __projection_type__ (CHOICE, required, default: cylindrical): One of `cylindrical`, `equirectangular`.

Hidden (standard ComfyUI): `prompt`, `extra_pnginfo`.

## Outputs
- None (output node). Saves JPEG files to ComfyUI output directory with XMP APP1 segment.

## Behavior
- Encodes image(s) to JPEG and injects an XMP block containing Google GPano metadata fields including `ProjectionType` and dimensions.
- Filenames include batch numbering and a counter.

## Usage Tips
- Use with stitched panorama or equirectangular sources to enable 360° viewer compatibility.
- Keep an eye on output quality; default is high (quality=95).

## Example
- `projection_type = cylindrical` for cylindrical panoramas.
- `projection_type = equirectangular` for 360×180 full panoramas.

## Version
- Introduced in StarNodes 1.6.0
