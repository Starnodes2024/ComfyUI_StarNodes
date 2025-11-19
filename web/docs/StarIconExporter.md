# ⭐ Star Icon Exporter

Exports a set of icons (PNGs and an ICO) from an input image, with optional shaping, padding, stroke, and shadow. Also produces a preview image batch.

- __Category__: `⭐StarNodes/Image And Latent`
- __Class__: `StarIconExporter`
- __File__: `star_icon_exporter.py`

## Inputs
- __images__ (IMAGE, required): Source image batch (first image is used).
- __save_name__ (STRING, required, default: "icon"): Base file name (no extension).
- __quantize_to_256__ (BOOLEAN, optional, default: true): Reduce to 256 colors (smaller files).
- __extra_sizes__ (STRING, optional, default: ""): Extra PNG sizes (comma‑separated, e.g. `64,512`).
- __subfolder__ (STRING, optional, default: "Icons"): Output subfolder under ComfyUI output.
- __shape__ (CHOICE, optional, default: none): `none`, `square`, `circle`, `rounded`.
- __rounded_radius_percent__ (INT, optional, default: 20): Corner radius percent for rounded.
- __background_color__ (STRING, optional, default: #FFFFFF): Background HEX for `square`.
- __padding_percent__ (INT, optional, default: 0): Uniform padding percent.
- __auto_trim__ (BOOLEAN, optional, default: true): Auto-trim transparent edges.
- __stroke_enabled__ (BOOLEAN, optional, default: false): Draw outline.
- __stroke_color__ (STRING, optional, default: #000000): Stroke color HEX.
- __stroke_width_percent__ (INT, optional, default: 6): Outline width percent.
- __shadow_enabled__ (BOOLEAN, optional, default: false): Add shadow.
- __shadow_color__ (STRING, optional, default: #000000): Shadow color HEX.
- __shadow_offset_px__ (INT, optional, default: 2): Shadow offset.
- __shadow_blur_px__ (INT, optional, default: 4): Shadow blur.
- __preset__ (CHOICE, optional, default: standard): `standard`, `windows`, `android`, `ios`, `web_favicon`.
- __naming_style__ (CHOICE, optional, default: increment): `increment`, `underscore_increment`, `timestamp`.
- __export_web_favicons__ (BOOLEAN, optional, default: false): Also export common web favicon PNG sizes.
- __preview_size__ (INT, optional, default: 256): Preview size to show in node.

## Outputs
- __ico_path__ (STRING): Path to the saved `.ico` file.
- __preview_batch__ (IMAGE): Preview of one selected size for the UI.

## Behavior
- Generates PNGs in `png/` and a combined `.ico` with selected sizes.
- Supports presets and additional custom sizes.
- Optional shaping (square/circle/rounded), stroke and shadow effects.
- Ensures non‑conflicting filenames based on `naming_style`.

## Usage Tips
- For platform packs, choose a `preset` (e.g., `windows`, `android`, `ios`).
- Use `shape = circle` or `rounded` to produce transparent background icons.
- Use `square` and set `background_color` to create solid background icons.

## Version
- Introduced in StarNodes 1.6.1
