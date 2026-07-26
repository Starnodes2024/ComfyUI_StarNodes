# ⭐ Star Save Image+

Save your images like the classic ComfyUI Save Image node — but modernized:
pick **Save or Preview** right on the node, save to **multiple formats at once**
(PNG, JPG, WEBP, PSD), embed an optional **mask**, and attach up to **5 custom
metadata fields**. This node is a drop-in replacement for the original
⭐ Star Save Image+ — old workflows keep working.

## On-node controls

- **💾 Save / 👁 Preview** — segmented control at the top of the node.
  - *Save* writes the files into your ComfyUI `output` folder (with all folder/naming options below).
  - *Preview* writes a temporary PNG (like the Preview Image node) — nothing is stored permanently and no metadata is embedded.
- **Format chips (PNG · JPG · WEBP · PSD)** — click to toggle. You can activate **several at once**; every active format is written with the same name and folder. At least one format always stays active. (PSD requires `pip install psd-tools`.)
- **Status line** — after running, shows exactly which files were written (hover for the full list).

## Inputs

- **images** — the image batch to save.
- **options** *(optional)* — connect a **⭐ Star Metadata Saver Option** node. Its 5 StarMetaData fields are embedded into every saved image.
- **mask** *(optional)* — embedded as alpha channel (**PNG/WEBP**) so **⭐ Star Load Image+** restores it automatically; **JPG** gets a `..._mask.png` sidecar; **PSD** gets a real layer mask.

## Widgets

- **preset_folder** — preset save folder from `presets.json` (overrides custom folder).
- **date_folder / date_folder_position** — add today's date as a folder, either as the first folder or as a subfolder.
- **custom_folder / custom_subfolder** — free-text folders.
- **date_in_filename** — add the date as filename prefix or suffix.
- **filename / add_timestamp / separator** — base filename, optional time stamp, and the separator used between parts.
- **jpg_quality / webp_quality / png_compress** — per-format quality settings.

## Outputs

- **path** — the save folder path relative to the ComfyUI output directory.

## Notes

- **No workflow data is embedded anymore** — only your 5 custom StarMetaData fields (PNG text chunks, EXIF for JPG/WEBP). PSD files do not carry metadata.
- Workflows saved with the original node migrate automatically: the old *save_jpg* switch turns into active PNG+JPG format chips.
- The 5 StarMetaData inputs are now on the **⭐ Star Metadata Saver Option** node — connect it to the **options** input to embed metadata.
- Read the metadata back with **⭐ Star Load Image+** + **⭐ Star Image Loader Options**.

## Related Nodes

- ⭐ Star Load Image+ (`StarLoadImagePlus`)
- ⭐ Star Metadata Saver Option (`StarMetadataSaverOption`)
- ⭐ Star Image Loader Options (`StarImageLoaderOptions`)
