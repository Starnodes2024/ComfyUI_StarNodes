# ⭐ Star Image Loader Options

Connect the **metadata** output of **⭐ Star Load Image+** and this node shows
every metadata entry stored in the image in a scrollable on-node list — each row
with a copy button — and gives you the **5 custom StarMetaData values** on their
own output connectors to reuse in your workflow.

## On-node list

After running the workflow, all metadata entries appear as key/value rows.
Custom fields show the label you gave them in ⭐ Star Metadata Saver Option.
Click **⧉** to copy any value to the clipboard.

## Inputs

- **metadata** — STAR_METADATA from ⭐ Star Load Image+.
- **lookup_key** — optional: type *any* metadata key or custom field name (exact, case-insensitive, or partial) and read its value from the **lookup_value** output.

## Outputs

| Output | Type | Content |
|---|---|---|
| StarMetaData 1–5 | STRING | The 5 custom metadata values |
| lookup_value | STRING | Value of the key typed into **lookup_key** |
| raw_json | STRING | All metadata as formatted JSON |

## Tips

- Missing entries output empty strings, so the node never breaks your workflow.
- The list shows *everything* found in the file — including entries written by other tools.

## Related Nodes

- ⭐ Star Load Image+ (`StarLoadImagePlus`)
- ⭐ Star Save Image+ (`StarSaveImagePlus`)
- ⭐ Star Metadata Saver Option (`StarMetadataSaverOption`)
