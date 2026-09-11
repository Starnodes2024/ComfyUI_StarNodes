# ⭐ Star Load Image+

Load an image from your ComfyUI `input` **or `output`** folder — or paste it
straight from the clipboard — and get back the image, its mask and the 5 custom
StarMetaData values stored by ⭐ Star Save Image+. Drop-in replacement for the
original ⭐ Star Load Image+ — old workflows keep working.

## On-node controls

- **📋 Paste Image** — paste an image directly from the clipboard (also available in the node's right-click menu). The pasted image is uploaded to the input folder automatically.
- **Metadata badge** — after running, shows how many metadata entries were found in the file.

## Inputs

- **image** — pick a file from the **input folder** *or* the **output folder** (entries ending with `[output]`, including subfolders like the date folders ⭐ Star Save Image+ creates), or use *Choose file to upload*. PNG, JPG, WEBP, BMP and GIF are supported.
- **invert_mask** — flip the mask extracted from the alpha channel (e.g. switch between inpaint-area and keep-area).

## Outputs

- **image** — the loaded image.
- **mask** — the alpha channel as a mask (empty mask when there is no alpha). Restores masks embedded by ⭐ Star Save Image+; flip it with **invert_mask**.
- **StarMetaData 1–5** — the 5 custom metadata values (same output slots as the original node).
- **metadata** — *all* metadata found in the file (STAR_METADATA). Connect this to **⭐ Star Image Loader Options** to see the full list and use single entries.

## Tips

- To read back embedded metadata, load the file from the **output** folder — *not* a temporary preview image. Temp previews don't contain the StarMetaData fields.
- The loader detects file changes automatically (re-runs when the file on disk changes).
- Legacy `StarMetaData 1…5` entries written by the original node are read exactly the same way.

## Related Nodes

- ⭐ Star Save Image+ (`StarSaveImagePlus`)
- ⭐ Star Image Loader Options (`StarImageLoaderOptions`)
- ⭐ Star Metadata Saver Option (`StarMetadataSaverOption`)
