# ⭐ Star Metadata Saver Option

Optional companion node for **⭐ Star Save Image+**. Provides **5 custom
metadata fields** (StarMetaData 1–5) that get embedded into every saved image.

Connect its **save_options** output to the **options** input of ⭐ Star Save Image+.

## Widgets

- **StarMetaData 1–5** — the values to embed. Only non-empty values are stored.

## How it is stored

- Values are embedded as `StarMetaData 1…5` text entries (PNG text chunks, EXIF for JPG/WEBP) — the same keys the original node used, so ⭐ Star Load Image+ and ⭐ Star Image Loader Options can read them.
- **No workflow/prompt data is embedded** — only your custom fields. PSD files do not carry metadata.

## Tips

- Typical uses: artist name, client/project, rating, notes, prompt variations you want to remember.
- Anything a text node can produce can be wired... simply type it or feed it in — the values are plain strings.

## Related Nodes

- ⭐ Star Save Image+ (`StarSaveImagePlus`)
- ⭐ Star Load Image+ (`StarLoadImagePlus`)
- ⭐ Star Image Loader Options (`StarImageLoaderOptions`)
