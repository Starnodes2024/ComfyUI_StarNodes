# ⭐ Star Dynamic LoRA

The **⭐ Star Dynamic LoRA** node lets you apply any number of LoRAs to a
MODEL and CLIP, using a compact, self-expanding DOM-panel UI with per-LoRA
toggle switches and a single strength slider.

- **Category**: `⭐StarNodes/Sampler`
- **Name**: `⭐ Star Dynamic LoRA`
- **Inputs (required)**:
  - `model` (`MODEL`)
  - `clip` (`CLIP`)
- **Dynamic parameters (LoRA slots)**:
  - `loraN_name` (combo, LoRA file or `None`)
  - `strengthN` (FLOAT, applied to both model and CLIP)
  - `enabledN` (BOOLEAN, on/off toggle)
- **Outputs**:
  - `model` (`MODEL` with all LoRAs applied)
  - `clip` (`CLIP` with all LoRAs applied)

The first slot (`lora1_name`, `strength1`, `enabled1`) is always present.
Additional slots are created dynamically.

## How the DOM-panel UI works

This node uses a JS helper (`web/js/star_dynamic_lora.js`) that renders a
styled DOM panel inside the node body (same pattern as Star Video Joiner
and the other StarNodes V2 widgets).

- Each LoRA slot is a single row with:
  - A **toggle dot** (●/○) to enable or disable the LoRA without clearing it.
  - A **dropdown** to pick the LoRA file.
  - A **strength slider** (-1 to 2 visual range) plus a numeric input box
    that accepts the full -100 to 100 range.
- When you pick a LoRA in the **last** row, a **new empty slot** is
  automatically added below.
- Trailing empty (`None`) slots are pruned automatically (at least one slot
  is always kept).
- The panel colors follow the active **StarNodes Theme** (set in ComfyUI
  settings → StarNodes Theme).

This makes it easy to stack multiple LoRAs for the same model without
having to chain multiple nodes.

---

# ⭐ Star Dynamic LoRA (Model Only)

The **⭐ Star Dynamic LoRA (Model Only)** node is a simpler variant that
only affects the MODEL (no CLIP).

- **Category**: `⭐StarNodes/Sampler`
- **Name**: `⭐ Star Dynamic LoRA (Model Only)`
- **Inputs (required)**:
  - `model` (`MODEL`)
- **Dynamic parameters (LoRA slots)**:
  - `loraN_name` (combo, LoRA file or `None`)
  - `strengthN` (FLOAT, model strength)
  - `enabledN` (BOOLEAN, on/off toggle)
- **Outputs**:
  - `model` (`MODEL` with all LoRAs applied)

The DOM-panel UI and auto-expand behavior are identical to the full
Dynamic LoRA node.

---

## Notes

- LoRA file list is taken from ComfyUI's `loras` folder.
- Any slot where `loraN_name = None`, `strengthN = 0`, or `enabledN = false`
  is ignored.
- LoRAs are applied in **slot index order** (1, 2, 3, ...).
- The single `strengthN` value is used for both model and CLIP (in the full
  node). This replaces the older separate `strengthN_model` / `strengthN_clip`
  sliders for a more compact UI.
