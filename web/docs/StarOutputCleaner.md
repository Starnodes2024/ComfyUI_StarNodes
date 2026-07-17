# ⭐ Star Output Cleaner

Browse, select and clean up the images in your ComfyUI **output folder**, or in a
**custom folder** of your choice (including all subfolders) — directly from a node,
with 200×200 px thumbnails and paging.

## Inputs

| Input | Type | Description |
|---|---|---|
| `source` | combo | `output folder` — browse the ComfyUI output directory.<br>`custom folder` — browse any folder you specify in the node's path field. |
| `mode` | combo | `last N` — images from the last N days/weeks/months.<br>`date range` — images between a start and an end date. |
| `custom_folder` | string | Folder for `custom folder` mode. Set it via the **path text field** or the **📂 Browse…** button inside the node — the picker can reach any folder on the machine. Saved with the workflow. |
| `amount` | int | How many days/weeks/months back to show (`last N` mode), `0` = all. Set via the **show last** number field inside the node. |
| `unit` | combo | `days` / `weeks` / `months` for `last N` mode (calendar months). Set via the dropdown inside the node. |
| `start_date` / `end_date` | string | `YYYY-MM-DD` bounds for `date range` mode (end is inclusive; empty = unbounded). Set via the **from / to** date pickers inside the node. |

Only `source` and `mode` appear as widgets on the node body — everything else is
edited via the controls inside the node's panel (they are still saved with the
workflow). The gallery reloads automatically whenever a value changes; use
**⟳ Refresh** to reload manually (e.g. after new images were generated).

## Node UI

- **Thumbnail grid** — the images matching the folder + date filter, newest first,
  as 200×200 px thumbnails. Hover for full path, file size and modification date.
- **Click** a thumbnail to toggle its checkbox (gold frame = selected).
  **Double-click** to open the **original image in full size** in a new tab.
- **Paging** — the grid shows one page at a time (default 100 images; choose
  50 / 100 / 200 / 500 in **per page**). Navigate with **◀ Prev / Next ▶**.
  Paging keeps thumbnail generation light on huge folders.
- **Select page** selects everything on the current page; **Select none** clears the
  whole selection. Selections are kept when you switch pages, and the counter shows
  the total number of selected images across all pages.
- **🗑 Delete selected** — permanently deletes all selected files (across all pages).
  A warning dialog first tells you that the images will be deleted **permanently**
  and **cannot be restored** — you must explicitly click **Delete permanently**
  (or **Cancel** / click outside to abort). *There is no undo and no recycle bin.*
- **📦 Download ZIP** — downloads all selected images (across all pages) as one ZIP
  file. Subfolder structure is preserved inside the archive.

## Outputs

None — this is a pure utility (output) node. Place it anywhere in the workflow;
no connections are required.

## Notes & safety

- Only files **inside the currently browsed root folder** (output dir or your custom
  folder) can be listed, zipped or deleted — path-traversal protection on every request.
- **Custom folder caution:** in `custom folder` mode the node can delete real files in
  that folder. Double-check the path, and be careful when loading workflows shared by
  others that use this mode.
- The folder picker lists subfolders anywhere on the server's drive; on Windows a
  **💻 Drives** button jumps back to the drive list. Hidden folders (starting with `.`)
  are skipped in the picker but can be typed into the path field manually.
- Thumbnails are generated once and cached in ComfyUI's **temp** directory
  (`temp/star_output_cleaner_thumbs`), so they never appear in the gallery itself.
  The cache refreshes automatically when a file changes.
- Temporary ZIP files are removed from the temp directory ~10 minutes after download.
- Supported image types: png, jpg, jpeg, webp, gif, bmp, tif, tiff.
- No extra Python dependencies are required (see `requirements.txt`).

## Troubleshooting

| Problem | Fix |
|---|---|
| Gallery is empty | Check the date filter (`amount = 0` shows everything) and press **⟳ Refresh**. |
| "custom folder path is empty" | `source` is `custom folder` but no path was entered. |
| "folder not found: …" | The custom folder path doesn't exist or isn't a folder — check for typos. |
| Node UI looks stale after an update | Restart ComfyUI **and** hard-refresh the browser (Ctrl/Cmd+Shift+R) to clear the cached JavaScript. |
| Old thumbnails after overwriting a file | The cache is keyed by modification time, so it updates automatically; press **⟳ Refresh**. |
| Delete seems to do nothing | Files in use by another program can fail to delete; the status line shows how many failed. |
