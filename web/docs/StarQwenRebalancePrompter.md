# ⭐ Star Qwen-Rebalance-Prompter

## Overview
The Star Qwen-Rebalance-Prompter creates structured, highly detailed prompts in the Qwen-Rebalance JSON format. This node helps you compose complex scenes by breaking them down into layers (foreground, midground, background) with precise composition, lighting, and color guidance.

## Category
`⭐StarNodes/Prompts`

## Inputs

### Required
- **subject** (STRING, multiline)
  - Main subject of the image
  - Example: "woman with short dark hair, backless white dress, barefoot, holding straw hat"

- **foreground** (STRING, multiline)
  - Elements in the front layer of the scene
  - Example: "wet rocks, shallow water, stone shoreline"

- **midground** (STRING, multiline)
  - Middle layer elements where main action occurs
  - Example: "woman standing on rocky shore, calm water surface, distant pier posts"

- **background** (STRING, multiline)
  - Background elements and scenery
  - Example: "coastal town with red-tiled roofs, church bell tower, hazy mountains"

- **composition_preset** (DROPDOWN)
  - Pre-defined composition styles:
    - Rule of thirds, off-center subject, vertical orientation, depth layering
    - Centered composition, symmetrical balance, horizontal orientation
    - Golden ratio, diagonal leading lines, dynamic perspective
    - Rule of thirds, horizontal orientation, balanced depth
    - Centered subject, radial composition, circular flow
    - Off-center, asymmetric balance, negative space emphasis
    - Custom (use custom_composition field)

- **custom_composition** (STRING)
  - Custom composition description (only used when "Custom" preset is selected)
  - Example: "spiral composition, leading eye from bottom-left to top-right"

- **color_tone** (STRING, multiline)
  - Color palette and tone description
  - Example: "soft pastel tones, muted earthy palette, isolated warm orange accents"

- **lighting_mood** (STRING, multiline)
  - Lighting conditions and mood
  - Example: "soft diffused daylight, gentle shadows, warm golden hour glow"

- **visual_guidance** (STRING, multiline)
  - Additional visual style guidance
  - Example: "photorealistic, high detail, natural textures, subtle depth of field"

- **caption** (STRING, multiline)
  - Overall scene caption or description
  - Example: "Serene coastal scene with woman contemplating by the shore"

## Outputs
- **prompt** (STRING)
  - JSON-formatted structured prompt containing all composition elements
  - Format: `{"subject": "...", "foreground": "...", "midground": "...", "background": "...", "composition": "...", "visual_guidance": "...", "color_tone": "...", "lighting_mood": "...", "caption": "..."}`

## Usage
1. Fill in each layer of your scene (subject, foreground, midground, background)
2. Select a composition preset or create a custom composition
3. Define color tone and lighting mood
4. Add visual guidance for style control
5. Provide an overall caption
6. Connect the output to your prompt processing nodes

## Tips
- **Layer Thinking**: Break your scene into distinct depth layers for better composition control
- **Composition Presets**: Use presets as starting points, then customize as needed
- **Visual Guidance**: Include technical details like "photorealistic", "painterly", "cinematic" for style control
- **Color Tone**: Be specific about color relationships and palette choices
- **Lighting Mood**: Lighting dramatically affects the final image mood

## Example Workflow
```
Star Qwen-Rebalance-Prompter → Text to Conditioning → Sampler
```

## Common Use Cases
- Complex scene composition with multiple depth layers
- Photorealistic image generation with precise control
- Architectural and landscape scenes
- Portrait photography with environmental context
- Cinematic scene composition

## Notes
- Output is JSON-formatted for compatibility with Qwen-based models
- All fields are combined into a structured prompt
- Empty fields are included but won't affect generation
- Works best with models trained on structured prompting

## Related Nodes
- Star Ollama Sysprompter (JC) - For AI-assisted prompt generation
- Star Image Edit for Qwen/Kontext - For Qwen-based image editing
- Star Qwen Regional Prompter - For region-specific prompting
- Star Qwen Edit Encoder - For Qwen conditioning
