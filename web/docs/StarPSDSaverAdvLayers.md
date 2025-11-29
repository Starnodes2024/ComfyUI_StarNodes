# Star PSD Saver Adv. Layers

## Description
This node saves multiple image layers with masks and **per-layer blend modes and opacity** to an optional Photoshop PSD file, and **always** outputs a flattened composited image.

It extends the original **Star PSD Saver (Dynamic)** with:
- Per-layer Photoshop-like blend modes
- Per-layer opacity (0–100%)
- A flattened `IMAGE` output
- A toggle to enable/disable writing the PSD file to disk

## Inputs

### Required
- **filename_prefix**  
  Base name for the output PSD file when saving is enabled.  
  Default: `"multilayer_adv"`

- **output_dir**  
  Subdirectory under the ComfyUI output folder where PSD files are saved.  
  Default: `"PSD_Layers"`

- **save_psd**  
  Boolean toggle to control whether a PSD file is written to disk.  
  - `True` (default): Save a PSD file with all layers, masks, blend modes, and opacity.  
  - `False`: **Do not** save a PSD file; only the flattened image is returned.

### Optional (dynamic)
- **layer1, layer2, layer3, ...**  
  Image layers (IMAGE type).  
  The JS extension automatically adds new `layerN` / `maskN` inputs as you connect more layers.

- **mask1, mask2, mask3, ...**  
  Optional masks (MASK type) corresponding to each `layerN`.

- **blend_mode2, blend_mode3, ...**  
  Blend mode for each layer **above** the base layer (layer1).  
  Supported modes include (Photoshop-like):
  - `normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, `color_dodge`, `color_burn`,  
    `soft_light`, `hard_light`, `vivid_light`, `linear_light`, `pin_light`, `hard_mix`,  
    `difference`, `exclusion`, `subtract`, `divide`, `linear_burn`, and more.

- **opacity2, opacity3, ...**  
  Opacity/strength for each layer above the base layer.  
  Range: `0`–`100` (%), default: `100`.

> **Note:** `blend_mode1` and `opacity1` are not exposed; the first layer acts as the base.

## Outputs

- **flattened_image** (IMAGE)  
  A single composited image of all layers, with their blend modes, opacity, and masks applied.

This output is always produced, even if `save_psd` is set to `False`.

## Usage

1. Connect one or more image nodes to `layer1`, `layer2`, etc.  
   - The UI will automatically add a new empty `layerN`/`maskN` pair when you connect the last one.
2. Optionally connect masks to `mask1`, `mask2`, etc.
3. For each layer above the first, adjust `blend_modeN` and `opacityN` to control how it mixes with the result below.
4. Set:
   - `filename_prefix`  
   - `output_dir`  
   - `save_psd` (toggle PSD saving on/off)
5. Run the workflow.  
   - If `save_psd = True`, a PSD file will be written to `ComfyUI/output/<output_dir>/...`  
   - The `flattened_image` output can be used directly in further image-processing nodes.

## Features

- **Per-layer blend modes** similar to Photoshop, controlled per dynamic layer.
- **Per-layer opacity** allows fine control over each layer's contribution.
- **Optional PSD saving** via `save_psd` to avoid disk writes when you only need the flattened result.
- **Dynamic inputs**: Additional `layerN`/`maskN` inputs appear automatically as you connect more layers.
- **Mask-aware compositing** so that each mask correctly affects its corresponding layer.
- **Flattened output** directly usable in ComfyUI workflows without needing to open the PSD.

## Notes

- Layer order is determined by the layer index: `layer1` at the bottom, `layer2` above it, etc.
- The PSD canvas size is determined by the largest connected image; smaller images are centered.
- When `save_psd = False`, no PSD file is written, but behavior of blending and masking for the
  `flattened_image` output remains identical.
