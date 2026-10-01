# ⭐ Star JSON Preview

## Overview

The **Star JSON Preview** node inspects a JSON string and renders its structure as a **collapsible, syntax-highlighted tree directly inside the node** — plus an **IMAGE output** that draws a layout mockup of the described scene. Connect any STRING output that contains JSON — a settings bundle, an API response, a file loader — run the workflow, and the node shows you exactly what is inside: objects, arrays, keys, values and nesting depth, without opening a text editor or copying the string elsewhere.

## Description

JSON strings produced by other nodes are often long and hard to read as raw text. This node parses the JSON and displays it as an expandable tree:

- **Objects** `{…}` and **arrays** `[…]` are collapsible — click a row to open or close it. A badge shows how many keys or items each level contains.
- **Values** are color-coded: strings, numbers, booleans and `null` each get their own color.
- A **status bar** summarizes the document: root type, number of objects/arrays/strings/numbers/booleans/nulls, maximum depth and size in bytes.
- **Invalid JSON** is shown with the exact parse error (message, line and column) instead of a silent failure.
- A small toolbar lets you **Expand all**, **Collapse all** and **Copy** the raw JSON to the clipboard.

Rendering is lazy — child rows are only built when you expand a level — so even large documents stay responsive. Levels with more than 1000 entries are truncated with a "… and N more" note.

The node is an `OUTPUT_NODE`, so it executes whenever its input changes even if nothing is connected downstream. The input string is also passed through unchanged on the `json` output, so you can place the node in the middle of a text chain.

## Inputs

### Required

- **json** (`STRING`, forced input)
  - Any string that contains JSON. Connect it from any node with a STRING output.
  - *Tooltip: "Connect any STRING that contains JSON. The structure is rendered as a collapsible tree inside the node and as a placement-box layout on the image output."*

- **aspect_ratio** (dropdown)
  - Frame ratio for the layout image — the **same list as ⭐ Starnodes Aspect Ratio Advanced** (1:1, 8:5, 4:3, 3:2, 7:5, 16:9, 21:9, 19:9 and the portrait variants), loaded from `json/sdratios.json`.
  - The layout image is rendered at **~2 megapixels** in the selected ratio (e.g. 16:9 → 1888×1064, 9:16 → 1064×1864), dimensions divisible by 8.

## Outputs

- **json** (`STRING`)
  - The connected string, passed through unchanged.
  - Useful for chaining: Preview → next node keeps receiving the original text.

- **info** (`STRING`)
  - A human-readable summary of the document, e.g.:
    ```
    Valid JSON
    Root: object — 5 keys
    Objects: 4 · Arrays: 2 · Strings: 18 · Numbers: 7 · Booleans: 1 · Nulls: 0
    Depth: 4 · Size: 1842 bytes
    ```
  - On invalid input: `Invalid JSON: Expecting property name enclosed in double quotes: line 3 column 5 (char 42)`
  - Connect to any STRING input, or use ⭐ Star Show Everything to inspect it.

- **image** (`IMAGE`)
  - A **layout mockup** of the described scene: every element that carries placement info is drawn as a colored box at its position on a frame in the selected `aspect_ratio` (~2 MP), labeled with its name and description, with a faint rule-of-thirds grid.
  - Placement is detected from either:
    - **Coordinates** — `x`, `y`, `w`, `h` (normalized 0–1 or pixels), `x1`/`y1`/`x2`/`y2`/`left`/`top`/`right`/`bottom`, or a 4-number array under `box` / `bbox` / `rect` / `bounds` / `coordinates` / `region` / `position` — the format emitted by the *Ideogram4 Image Prompt Refiner (JSON)* system prompt in `json/startext.json`.
    - **Region text** — natural language like `"top left"`, `"bottom right corner"`, `"center"`, `"background"`, `"foreground"`, `"across the top"`, `"left side"`.
  - Box colors come from each element's own `color_palette` (first hex color), falling back to a rotating palette. `type: "text"` elements show their quoted phrase as the box title.
  - If the document has a `style_description.color_palette`, the swatches are drawn as a strip in the bottom-left corner.
  - **Fallback:** JSON without any placement info renders as a nested colored-box *structure* diagram instead (purple = object, teal = array, green = string, orange = number, pink = boolean, gray = null).
  - On invalid input the image shows a red box with the parse error.
  - Very large documents are kept readable: rendering stops after ~600 boxes (deep or crowded branches collapse into gray `… N more` stubs, depth is capped at 8 levels, each level shows at most 20 children, long texts are shortened).

## Tips

- The tree only renders after the node has **executed** — press *Queue* (or run the part of the workflow feeding it) to refresh the preview.
- **Expand all** opens every level, including deeply nested ones; on very large documents prefer expanding single branches.
- Long scalar values are shortened in the display with a `… (N more chars)` note — the `json` output always carries the full original text.

---

**Category**: `⭐StarNodes/Text And Data`

**Node name**: `StarJsonPreview`

**Display name**: `⭐ Star JSON Preview`
