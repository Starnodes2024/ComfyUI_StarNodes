# ⭐ Star Image Edit for Qwen/Kontext

A prompt builder node that outputs instruction strings tailored for the two image-editing models:

- Qwen Image Edit (`Qwen/Qwen-Image-Edit`)
- FLUX.1 Kontext (Dev/Pro/Max)

It reads task templates from `editprompts.json` so you can easily add or modify tasks without changing code.

## Inputs

- model: Select "Qwen Image Edit" or "FLUX.1 Kontext"
- task: One of the common edit tasks loaded from `editprompts.json`
- subject, color, background, object, location, style, surface, text, lighting, expression, clothing_item, style_or_color: Generic fields used by templates. Fill the ones relevant to your chosen task.
- keep_clause (optional): Add your own or additional prompt instructions.

## Outputs

- prompt (STRING): The assembled instruction string
- params_hint (STRING): A small hint showing which placeholders the selected task expects and which are missing.

## Usage Tips

- Qwen Image Edit accepts straightforward edit instructions, e.g.:
  - "Change the rabbit's color to purple."
  - "Replace the background with a sunset sky."
- FLUX.1 Kontext responds best to precise action verbs and explicit subjects:
  - "Replace the background with a neon city at night. Keep the woman in the red dress unchanged."
  - "Change the jacket to black leather on the man with glasses. Keep face and identity unchanged."
- For text editing, specify the surface and quote the new text: "Replace the text on the storefront sign with \"Open 24/7\"."

## Customizing Tasks (editprompts.json)

The file `editprompts.json` defines models and tasks. Structure example:

```json
{
  "version": 1,
  "models": {
    "Qwen Image Edit": {
      "tasks": {
        "Change Color": {
          "description": "Change the color of a specific object or region.",
          "template": "Change the {subject}'s color to {color}.",
          "params": ["subject", "color"]
        }
      }
    },
    "FLUX.1 Kontext": {
      "tasks": {
        "Change Clothing": {
          "description": "Alter clothing while keeping identity.",
          "template": "Change the {clothing_item} to {style_or_color} on the {subject}. Keep the person's face and identity unchanged.",
          "params": ["clothing_item", "style_or_color", "subject"]
        }
      }
    }
  }
}
```

## Example Workflows

- Use this node to build the prompt, then feed to your encoder or pipeline:
  - For Qwen: combine with `⭐ Star Qwen Edit Encoder` (`star_qwen_edit_encoder.py`) and `⭐ Star Qwen Image Edit Inputs` (`star_qwen_image_edit_inputs.py`) to prepare conditioning and resolution.
  - For FLUX Kontext: feed the prompt into your Kontext runner node or API integration.

## Notes

- The node hot-reloads `editprompts.json` on modification time change.
- If a template is missing or fields are blank, the output prompt will indicate what to fill.
