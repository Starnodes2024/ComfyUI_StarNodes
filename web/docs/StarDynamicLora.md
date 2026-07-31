# ⭐ Star Dynamic LoRA

The **⭐ Star Dynamic LoRA** node lets you apply any number of LoRAs to a
MODEL and CLIP, using a compact, expandable UI.

- **Category**: `⭐StarNodes/Sampler`
- **Name**: `⭐ Star Dynamic LoRA`
- **Inputs (required)**:
  - `model` (`MODEL`)
  - `clip` (`CLIP`)
- **Dynamic parameters (LoRA slots)**:
  - `loraN_name` (combo, LoRA file or `None`)
  - `strengthN_model` (FLOAT, model strength)
  - `strengthN_clip` (FLOAT, CLIP strength)
- **Outputs**:
  - `model` (`MODEL` with all LoRAs applied)
  - `clip` (`CLIP` with all LoRAs applied)

The first slot (`lora1_name`, `strength1_model`, `strength1_clip`) is
always present. Additional slots are created dynamically.

## How the dynamic UI works

This node uses a small JS helper (`web/js/star_dynamic_lora.js`) to
manage dynamic LoRA slots.

- On node creation, the first slot is ensured:
  - `lora1_name`, `strength1_model`, `strength1_clip`.
- When you change the **last** `loraN_name` to something other than
  `None`, a **new empty slot** is automatically added.
- The node context menu contains:
  - **Add LoRA**: force-add another LoRA slot.
  - **Remove Last LoRA**: remove the last slot (first slot is kept).

This makes it easy to stack multiple LoRAs for the same model without
having to chain multiple nodes.

---

# ⭐ Star Dynamic LoRA (Model Only)

The **⭐ Star Dynamic LoRA (Model Only)** node is a simpler variant that
only affects the MODEL (no CLIP strength).

- **Category**: `⭐StarNodes/Sampler`
- **Name**: `⭐ Star Dynamic LoRA (Model Only)`
- **Inputs (required)**:
  - `model` (`MODEL`)
- **Dynamic parameters (LoRA slots)**:
  - `loraN_name` (combo, LoRA file or `None`)
  - `strengthN_model` (FLOAT, model strength)
- **Outputs**:
  - `model` (`MODEL` with all LoRAs applied)

The behavior of adding/removing slots is identical to the full Dynamic
LoRA node, but each slot only has **model strength**.

---

## Notes

- LoRA file list is taken from ComfyUI's `loras` folder.
- Any slot where `loraN_name = None` or `strengthN_model = 0` is
  ignored.
- LoRAs are applied in **slot index order** (1, 2, 3, ...).
