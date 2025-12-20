# ⭐ Star Load Image+

Load images like ComfyUI's default **Load Image** node, but with **5 additional text outputs** populated from the image metadata.

- __Category__: `⭐StarNodes/IO`
- __Class__: `StarLoadImagePlus`
- __File__: `image_tools/star_load_image_plus.py`

## Overview

This node extracts the following metadata keys from the loaded image:
- `StarMetaData 1`
- `StarMetaData 2`
- `StarMetaData 3`
- `StarMetaData 4`
- `StarMetaData 5`

If a key is not present, the output is an empty string.

---

## Inputs

- __image__ (STRING / image upload, required): Image file from your ComfyUI input folder.

## Outputs

- __image__ (IMAGE): Loaded image tensor.
- __mask__ (MASK): Alpha mask if present, otherwise empty mask.

### Star metadata fields

- __StarMetaData 1__ (STRING)
- __StarMetaData 2__ (STRING)
- __StarMetaData 3__ (STRING)
- __StarMetaData 4__ (STRING)
- __StarMetaData 5__ (STRING)

## Notes

- Works best with images saved by **Star Save Image+**, but will also output values if any other tool stored these keys in metadata.

## Related Nodes

- ⭐ Star Save Image+ (`StarSaveImagePlus`)
- ⭐ Star Meta Injector (`StarMetaInjector`)
