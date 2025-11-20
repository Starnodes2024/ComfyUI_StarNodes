# ⭐ Star Ollama Sysprompter (JC)

Builds two prompt strings to use with Ollama or any LLM: a System Prompt and a Detail Prompt.

This node lets you choose a style from a dropdown (loaded from `styles.json`) or type your own. It also appends an optional "Additional System Prompt" to the system message.

## Inputs

- **max_tokens (INT)**: Maximum tokens to aim for in the detail prompt where the `###` placeholder is used. Default: `300`.
- **style (CHOICE)**: Dropdown populated from `styles.json`. The first entry is `Own Style`.
- **own_style (STRING)**: If `Own Style` is chosen, this text is used for both `STYLENAME` and `STYLE`.
- **additional_system_prompt (STRING)**: Appended to the end of the system prompt if not empty.
- **Fit Composition to Style (BOOLEAN)**: When enabled, appends `Change image composition to fit the chosen style.` to the system prompt.

## Outputs

- **system_prompt (STRING)**: Fixed, plus your additional system prompt (if provided).
- **detail_prompt (STRING)**: Uses your token count and chosen style name/description.
- **style_name (STRING)**: The resolved style display name (selected dropdown value or your own style string).

## Behavior details

- When `style` is not `Own Style`, `STYLENAME` is taken from the selected style's `name`, and `STYLE` from its `style` description.
- When `style` is `Own Style`, your `own_style` string is used for both `STYLENAME` and `STYLE`.
- `max_tokens` replaces the placeholder `###` in the detail prompt line.
- If `Fit Composition to Style` is enabled, the system prompt will include: `Change image composition to fit the chosen style.`

## Prompt templates

- **System Prompt** (fixed):
  `You are an AI artist. you create one image prompt. no questions or comments. [Additional System Prompt if provided]`

- **Detail Prompt**:
  `describe the image and create an image prompt with ### tokens max. no questions or comments. change style to *STYLENAME*. turn prompt into *STYLE*`

In practice, the node outputs the detail prompt with `###` replaced by `max_tokens`, `*STYLENAME*` replaced by the selected or own style name, and `*STYLE*` replaced by the style description.

## styles.json

- Location: `custom_nodes/ComfyUI_StarBetaNodes/styles.json`
- Format: an array of objects with `name` and `style` keys, e.g.:

```json
[
  { "name": "Pencil Sketch", "style": "detailed pencil sketch" },
  { "name": "Photorealistic", "style": "highly detailed photorealistic" }
]
```

You can edit this file and reload custom nodes in ComfyUI to update the dropdown.

## Example

- Inputs: `max_tokens=300`, `style=Pencil Sketch`, `additional_system_prompt="Use Lora:metal-edges"`, `Fit Composition to Style=true`
- Outputs:
  - system_prompt:
    `You are an AI artist. you create one image prompt. no questions or comments. Use Lora:metal-edges Change image composition to fit the chosen style.`
  - detail_prompt:
    `describe the image and create an image prompt with 300 tokens max. no questions or comments. change style to Pencil Sketch. turn prompt into detailed pencil sketch`
  - style_name:
    `Pencil Sketch`
