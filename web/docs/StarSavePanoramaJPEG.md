# ⭐ Star Save Panorama JPEG

Saves images as JPEG with embedded XMP panorama metadata for panorama viewers using cylindrical projection.

- __Category__: `⭐StarNodes/Image And Latent`
- __Class__: `StarSavePanoramaJPEG`
- __File__: `star_save_panorama_jpeg.py`

## Inputs
- __images__ (IMAGE, required): Batch of images to save.
- __filename_prefix__ (STRING, required, default: "ComfyUI"): Prefix for saved files.

Hidden (standard ComfyUI): `prompt`, `extra_pnginfo`.

## Outputs
- None (output node). Saves JPEG files to ComfyUI output directory with XMP APP1 segment.

## Behavior
- Encodes image(s) to JPEG and injects an XMP block containing Google GPano metadata fields with cylindrical projection type and image dimensions.
- Filenames include batch numbering and a counter.
- Uses cylindrical projection for panorama metadata (optimized for horizontal panoramas).

## Usage Tips
- Use with stitched cylindrical panoramas to enable panorama viewer compatibility.
- Output quality is set to high (quality=95) for best results.
- Compatible with Google Photos, Facebook 360, and other panorama viewers that support cylindrical projection.

## Example
- Connect panorama images to this node to save with proper cylindrical panorama metadata.
- Ideal for horizontal panoramic shots and wide-angle compositions.

## Version
- Introduced in StarNodes 1.6+
- Updated in StarNodes 1.8.0: Simplified to use only cylindrical projection
