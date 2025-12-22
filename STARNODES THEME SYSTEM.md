# StarNodes Theme System

This document explains how to use the StarNodes theme system in ComfyUI.

## What it does

StarNodes can style node colors (background + title bar) and draw an optional 1px border/frame around nodes.

You can:

- Pick a ready-to-use theme from ComfyUI Settings.
- Apply a theme preset from a node right-click menu.
- Apply a theme preset to multiple nodes at once (multi-select).
- Customize single nodes (background, title bar, frame color, frame width).
- Reset nodes back to ComfyUI/LiteGraph defaults.

## Enable and choose a theme (ComfyUI Settings)

1. Open ComfyUI.
2. Open **Settings**.
3. Find the **StarNodes** section.
4. Use:

- `StarNodes Theme` to choose a ready-to-use theme.
- `Apply StarNodes Style to All Nodes` if you want StarNodes styling applied to every node type.

Notes:

- If you enable `Apply StarNodes Style to All Nodes`, ComfyUI may require a page reload before the styling hooks apply to every node type.
- Changing the theme dropdown applies immediately to nodes currently loaded on the canvas.

## Apply a theme from the node right-click menu

1. Right click a node.
2. Choose: `⭐ Theme Presets`
3. Pick a theme.

### Multi-select support

If you select multiple nodes (box select / shift-click), then choosing a theme preset will apply it to **all selected nodes**.

## Customize a single node

Right click a node and use:

- `⭐ Change Color` (node background)
- `⭐ Title Bar` (title bar color)
- `⭐ Frame Color` (border color)
- `⭐ Frame Width` (border width in pixels)

These are stored on the node as per-node overrides, so they will remain even if you change the global theme.

## Reset / remove overrides

### Reset one node

Right click the node and use:

- `⭐ Reset Color` (clears background + title bar overrides)
- `⭐ Reset Frame` (clears border overrides)

### Reset using Theme Presets

Right click a node and choose:

- `⭐ Theme Presets` -> `ComfyUI Default (Clear Overrides)`

This clears StarNodes appearance overrides for that node (or all selected nodes).

### Reset ALL nodes from Settings

In ComfyUI Settings (StarNodes section) click:

- `Reset All Nodes to ComfyUI Defaults`

This clears StarNodes overrides from all nodes currently loaded in the canvas and switches the theme to the default (no overrides).

## Troubleshooting

- If you don’t see the theme dropdown/button in Settings, hard-refresh the browser (Ctrl+F5).
- If you enabled `Apply StarNodes Style to All Nodes` and it doesn’t affect non-StarNodes, reload the page.
- Per-node overrides always win over global theme defaults.
