# ⭐ Star Save Image+

Save images like ComfyUI's default **Save Image** node, but with:
- **5 connector-only string inputs** that get embedded into the image metadata
- built-in folder/filename settings (like the "Star Save Folder String" node)

- __Category__: `⭐StarNodes/IO`
- __Class__: `StarSaveImagePlus`
- __File__: `image_tools/star_save_image_plus.py`

## Overview

This node is meant for workflows where you want to save an image and also store extra workflow-related strings inside the image file.

It writes the following PNG metadata keys:
- `StarMetaData 1`
- `StarMetaData 2`
- `StarMetaData 3`
- `StarMetaData 4`
- `StarMetaData 5`

These can later be retrieved using **Star Load Image+**.

---

## Inputs

- __images__ (IMAGE, required): The images to save.

### Folder / filename settings

- __preset_folder__ (STRING): Preset base folder
- __date_folder__ (BOOLEAN): Put outputs into a date folder
- __date_folder_position__ (STRING): Where to place the date folder
- __custom_folder__ (STRING): Custom base folder
- __custom_subfolder__ (STRING): Custom subfolder
- __date_in_filename__ (STRING): Add date to filename (prefix/suffix)
- __filename__ (STRING): Base filename
- __add_timestamp__ (BOOLEAN): Add time stamp to filename
- __separator__ (STRING): Separator used for timestamp

### Star metadata fields (connector-only)

- __StarMetaData 1__ (STRING): Saved as `StarMetaData 1`
- __StarMetaData 2__ (STRING): Saved as `StarMetaData 2`
- __StarMetaData 3__ (STRING): Saved as `StarMetaData 3`
- __StarMetaData 4__ (STRING): Saved as `StarMetaData 4`
- __StarMetaData 5__ (STRING): Saved as `StarMetaData 5`

## Outputs

This is an output node that saves images directly and shows them in the UI.

## Notes

- The StarMetaData values are stored as plain text metadata.
- If ComfyUI metadata is enabled, the image will also contain the default ComfyUI metadata (prompt/workflow), just like the normal Save Image node.

## Related Nodes

- ⭐ Star Load Image+ (`StarLoadImagePlus`)
- ⭐ Star Meta Injector (`StarMetaInjector`)
