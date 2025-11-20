# ⭐ Star Qwen Regional Prompter

## Overview

**Star Qwen Regional Prompter** is a simplified ComfyUI node designed for **Qwen2-VL CLIP + Qwen-Image** workflows. It automatically divides your image into 4 equal quadrants, making regional prompting easy and intuitive.

## ✅ Designed For

- ✅ **Qwen2.5-VL CLIP** (qwen_2.5_vl_7b) for text encoding
- ✅ **Qwen-Image (MMDiT)** for image generation
- ✅ Uses **combined mode** (spatial descriptions + grounding tokens) - tested to work best!
- ❌ Not compatible with standard CLIP models (SD1.5, SDXL, etc.)

## Key Features

- 🎯 **Simple 4-Quadrant Layout**: Automatic division into Upper Left, Upper Right, Lower Left, Lower Right
- ⚡ **No Manual Coordinates**: Just enter prompts for the quadrants you want to control
- 🔄 **Optimized for Qwen-Image**: Uses combined mode (spatial + grounding tokens) for best results
- 🎨 **Background + Quadrants**: Set overall scene with background, control specific areas with quadrants
- 🧩 **Flexible**: Leave quadrants empty to skip them - only use what you need

## How It Works

### Automatic Quadrant Division

Your image is automatically split into 4 equal parts:

```
┌─────────────┬─────────────┐
│             │             │
│  Upper Left │ Upper Right │
│             │             │
├─────────────┼─────────────┤
│             │             │
│  Lower Left │ Lower Right │
│             │             │
└─────────────┴─────────────┘
```

For a 1024x1024 image:
- **Upper Left**: (0, 0) to (512, 512)
- **Upper Right**: (512, 0) to (1024, 512)
- **Lower Left**: (0, 512) to (512, 1024)
- **Lower Right**: (512, 512) to (1024, 1024)

### Combined Mode Formatting

Each quadrant is formatted with both spatial descriptions AND grounding tokens:

```
Input:
  Background: "Beautiful landscape"
  Upper Left: "mountains"
  Lower Left: "lake"

Output:
  "Beautiful landscape, mountains in the upper left area <|object_ref_start|>mountains<|object_ref_end|> <|box_start|>(0,0),(499,499)<|box_end|>, lake in the lower left area <|object_ref_start|>lake<|object_ref_end|> <|box_start|>(0,500),(499,999)<|box_end|>"
```

## Usage Guide

### Quick Start

1. **Add the node** to your ComfyUI workflow
2. **Connect Qwen2.5-VL CLIP** to the clip input
3. **Set image dimensions** (e.g., 1024x1024)
4. **Write background prompt**: "Professional photograph, cinematic lighting"
5. **Fill quadrants** (only the ones you want to control):
   - Region Upper Left: "mountains"
   - Region Upper Right: "mountains"
   - Region Lower Left: "lake"
   - Region Lower Right: "lake"
6. **Connect to Qwen-Image** for generation

### Simple Example

```
CLIP: Qwen2.5-VL-7B
Background: "Beautiful nature scene, golden hour"
Image: 1024x1024

Region Upper Left: "snow-capped mountain peaks"
Region Upper Right: "snow-capped mountain peaks"
Region Lower Left: "crystal clear alpine lake"
Region Lower Right: "crystal clear alpine lake"
```

**Result:** Mountains on top, lake on bottom!

### Example Use Cases

#### 1. Four Elements Composition
```
Background: "Fantasy magical scene, vibrant colors"
Image: 1024x1024

Region Upper Left: "fire element, flames and lava"
Region Upper Right: "water element, ocean waves"
Region Lower Left: "earth element, rocks and crystals"
Region Lower Right: "air element, clouds and wind"
```

#### 2. Day/Night Split
```
Background: "Artistic composition, surreal atmosphere"
Image: 1024x1024

Region Upper Left: "bright sunny day, blue sky"
Region Upper Right: "dark stormy night, lightning"
Region Lower Left: (empty)
Region Lower Right: (empty)
```

#### 3. Simple Top/Bottom Split
```
Background: "Nature photograph, professional quality"
Image: 1024x1024

Region Upper Left: "dramatic mountain peaks"
Region Upper Right: "dramatic mountain peaks"
Region Lower Left: "serene lake with reflections"
Region Lower Right: "serene lake with reflections"
```

## Parameters Reference

### Required Inputs

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `clip` | CLIP | - | - | Qwen2.5-VL CLIP model |
| `background_prompt` | STRING | - | "A beautiful scene" | Overall scene, style, and mood |
| `image_width` | INT | 64-8192 | 1024 | Target image width (pixels) |
| `image_height` | INT | 64-8192 | 1024 | Target image height (pixels) |

### Optional Inputs (Quadrants)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region_upper_left` | STRING | "" | Upper left quadrant prompt (empty = skip) |
| `region_upper_right` | STRING | "" | Upper right quadrant prompt (empty = skip) |
| `region_lower_left` | STRING | "" | Lower left quadrant prompt (empty = skip) |
| `region_lower_right` | STRING | "" | Lower right quadrant prompt (empty = skip) |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `CONDITIONING` | CONDITIONING | Optimized conditioning for Qwen-Image |

## Tips & Best Practices

### ✅ Do's

- **Match dimensions**: Set `image_width` and `image_height` to your actual Qwen-Image generation size
- **Be specific**: Use detailed descriptions - "snow-capped mountain peaks" not just "mountains"
- **Use background for style**: "Professional photograph, cinematic lighting, 8k quality"
- **Combine quadrants**: Want top half? Fill both upper quadrants with the same prompt
- **Leave empty**: Only fill quadrants you need - empty ones follow background prompt
- **Test simple first**: Try "red" vs "blue" vs "green" vs "yellow" to see regional control

### ❌ Don'ts

- **Don't use wrong CLIP**: Only works with Qwen2-VL CLIP models
- **Don't expect pixel-perfect**: Regional control is guidance, some blending is normal
- **Don't over-complicate**: Start with 1-2 quadrants, add more as needed
- **Don't mismatch dimensions**: Image size must match your actual generation settings

## Technical Details

### Coordinate Normalization Formula

```python
normalized_coord = int((pixel_coord / image_dimension) * 1000)
# Clamped to [0, 999]
```

### Bounding Box Calculation

```python
x2 = x + width  # Bottom-right X
y2 = y + height # Bottom-right Y

# Ensure within bounds
x2 = min(x2, image_width)
y2 = min(y2, image_height)
```

### Token Format

```
<|object_ref_start|>description<|object_ref_end|> <|box_start|>(x1,y1),(x2,y2)<|box_end|>
```

## Troubleshooting

### Issue: Regions not being respected

**Solutions:**
- Verify you're using Qwen2.5-VL CLIP model
- Check coordinates are within image bounds
- Ensure regional prompts are not empty
- Confirm image_width/height match your generation size

### Issue: Unexpected region placement

**Solutions:**
- Remember: origin is top-left (0,0)
- Check your coordinate calculations
- Verify width/height are positive values
- Use smaller test images to debug positioning

### Issue: Weak regional control

**Solutions:**
- Make regional prompts more specific and detailed
- Increase contrast between regional and background prompts
- Try non-overlapping regions first
- Experiment with different prompt strengths

## Research Background

This node is based on the official Qwen2-VL paper's visual grounding capabilities:

> "To endow the model with visual grounding capabilities, bounding box coordinates are normalized within [0, 1000) and represented as (X_topleft, Y_topleft), (X_bottomright, Y_bottomright). Tokens <|box_start|> and <|box_end|> are utilized to demarcate bounding box text. To accurately link bounding boxes with their textual descriptions, we introduce tokens <|object_ref_start|> and <|object_ref_end|>..."

**Source**: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/html/2409.12191v1)

## Version History

- **v2.0** (2025-01-11): Simplified quadrant-based interface
  - Automatic 4-quadrant division (equal sizes)
  - Removed manual coordinate inputs for ease of use
  - Fixed to combined mode (spatial + grounding tokens) - tested best for Qwen-Image
  - Renamed regions: Upper Left, Upper Right, Lower Left, Lower Right
  - Much simpler and more intuitive!
  
- **v1.0** (2025-01-11): Initial release
  - Support for background + 4 regional prompts with manual coordinates
  - Multiple prompt modes (spatial, grounding, combined)
  - Automatic coordinate normalization
  - Qwen2-VL special token formatting

## License

Part of ComfyUI_StarBetaNodes. See main repository for license information.

## Contributing

Found a bug or have a feature request? Please open an issue on the main repository.

## Credits

- Developed for ComfyUI_StarBetaNodes
- Based on Qwen2-VL visual grounding research by Alibaba
- Special thanks to the Qwen team for the excellent model and documentation

---

**Category**: ⭐StarNodes/Conditioning  
**Node Name**: StarQwenRegionalPrompter  
**Display Name**: ⭐ Star Qwen Regional Prompter
