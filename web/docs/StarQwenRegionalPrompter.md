# ⭐ Star Qwen Regional Prompter

A simplified regional prompting node designed for **Qwen2-VL CLIP + Qwen-Image** workflows. Automatically divides your image into 4 equal quadrants for easy regional control.

Category: `⭐StarNodes/Conditioning`

## Overview

This node makes regional prompting simple by automatically splitting your image into 4 equal quadrants:

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

Just enter prompts for the quadrants you want to control - leave others empty to skip them.

### How It Works

Uses **combined mode** (spatial descriptions + Qwen2-VL grounding tokens) which has been tested to work best with Qwen-Image generation. The node automatically:
1. Divides your image into 4 equal parts
2. Adds spatial descriptions ("in the upper left area", etc.)
3. Includes Qwen2-VL grounding tokens with normalized coordinates
4. Combines everything into optimized conditioning for Qwen-Image

## Inputs

### Required
- **clip (CLIP)**: The Qwen2.5-VL CLIP model used for conditioning
- **background_prompt (STRING)**: The base prompt describing the overall scene, style, and mood
- **image_width (INT)**: Width of the target image in pixels. Range: 64-8192, default: 1024
- **image_height (INT)**: Height of the target image in pixels. Range: 64-8192, default: 1024

### Optional (4 Quadrants)

Simply enter text for the quadrants you want to control. Leave empty to skip.

- **region_upper_left (STRING)**: Prompt for the upper left quadrant
- **region_upper_right (STRING)**: Prompt for the upper right quadrant  
- **region_lower_left (STRING)**: Prompt for the lower left quadrant
- **region_lower_right (STRING)**: Prompt for the lower right quadrant

## Outputs
- **CONDITIONING**: Optimized conditioning for Qwen-Image with regional control

## How It Works

### Automatic Quadrant Division

For a 1024x1024 image:
- **Upper Left**: (0, 0) to (512, 512)
- **Upper Right**: (512, 0) to (1024, 512)
- **Lower Left**: (0, 512) to (512, 1024)
- **Lower Right**: (512, 512) to (1024, 1024)

### Prompt Construction

Each quadrant with a prompt gets formatted as:
```
[prompt text] in the [position] area <|object_ref_start|>[prompt text]<|object_ref_end|> <|box_start|>(normalized_coords)<|box_end|>
```

**Example:**
- Input: `region_upper_left = "snow-capped mountains"`
- Output: `"snow-capped mountains in the upper left area <|object_ref_start|>snow-capped mountains<|object_ref_end|> <|box_start|>(0,0),(499,499)<|box_end|>"`

### Final Encoding

All quadrants are combined with the background prompt and encoded by Qwen2-VL CLIP for Qwen-Image generation.

## Usage Examples

### Example 1: Landscape Composition
```
Background: "Beautiful nature photograph, golden hour lighting, professional quality"
Image: 1024x1024

Region Upper Left: "dramatic mountain peaks with snow"
Region Upper Right: "dramatic mountain peaks with snow"  
Region Lower Left: "serene alpine lake with reflections"
Region Lower Right: "serene alpine lake with reflections"
```

**Result:** Mountains fill the top half, lake fills the bottom half.

### Example 2: Split Scene
```
Background: "Artistic composition, high contrast"
Image: 1024x1024

Region Upper Left: "bright sunny day with blue sky"
Region Upper Right: "dark stormy night with lightning"
Region Lower Left: (empty)
Region Lower Right: (empty)
```

**Result:** Day/night split in the upper half, background prompt controls the lower half.

### Example 3: Four Distinct Areas
```
Background: "Fantasy landscape, magical atmosphere"
Image: 1024x1024

Region Upper Left: "fire element, flames and lava"
Region Upper Right: "water element, ocean waves"
Region Lower Left: "earth element, rocks and crystals"
Region Lower Right: "air element, clouds and wind"
```

**Result:** Four elemental quadrants with distinct characteristics.

## Tips & Best Practices

1. **Use Descriptive Background Prompts**
   - Set overall style, mood, and quality in the background
   - Example: "Professional photograph, cinematic lighting, 8k quality"

2. **Be Specific in Quadrant Prompts**
   - Good: "snow-capped mountain peaks with detailed textures"
   - Bad: "mountains"

3. **Combine Quadrants for Larger Areas**
   - Want top half? Fill both upper quadrants with the same prompt
   - Want left half? Fill both left quadrants with the same prompt

4. **Leave Quadrants Empty**
   - Only fill the quadrants you need to control
   - Empty quadrants will follow the background prompt

5. **Match Image Dimensions**
   - Set `image_width` and `image_height` to your actual generation size
   - This ensures accurate quadrant division

6. **Test with Simple Prompts First**
   - Try basic concepts to see how regional control works
   - Example: "red color" vs "blue color" in different quadrants

## Compatibility

- **Designed for**: Qwen2.5-VL CLIP + Qwen-Image workflows
- **Required CLIP**: Qwen2.5-VL CLIP model (qwen_2.5_vl_7b or similar)
- **Required Diffusion Model**: Qwen-Image (MMDiT)
- **Not compatible with**: Standard CLIP models (SD1.5, SDXL, etc.)

## Technical Details

### Automatic Quadrant Calculation
```python
half_width = image_width // 2
half_height = image_height // 2

# Quadrants:
# Upper Left:  (0, 0) to (half_width, half_height)
# Upper Right: (half_width, 0) to (image_width, half_height)
# Lower Left:  (0, half_height) to (half_width, image_height)
# Lower Right: (half_width, half_height) to (image_width, image_height)
```

### Coordinate Normalization
Qwen2-VL expects coordinates in [0, 1000) range:
```python
normalized_coord = int((pixel_coord / image_dimension) * 1000)
# Clamped to [0, 999]
```

### Combined Mode Format
Each quadrant is formatted as:
```
[prompt] in the [position] area <|object_ref_start|>[prompt]<|object_ref_end|> <|box_start|>(x1,y1),(x2,y2)<|box_end|>
```

## Limitations

1. Fixed 4-quadrant layout (equal sizes)
2. Rectangular regions only
3. Regional control is guidance, not absolute positioning
4. Effectiveness depends on Qwen-Image's interpretation

## Troubleshooting

**Q: Quadrants are still mixing**
- Make quadrant prompts very distinct and specific
- Use the background prompt for overall style/mood only
- Try simple test: "red" vs "blue" vs "green" vs "yellow" in each quadrant
- Regional control is guidance - some blending is normal

**Q: How do I control just the top half?**
- Fill both `region_upper_left` and `region_upper_right` with the same prompt
- Leave lower quadrants empty

**Q: How do I control just one corner?**
- Fill only that quadrant's prompt
- Leave the other three empty

**Q: Image dimensions don't match my generation**
- Set `image_width` and `image_height` to match your actual Qwen-Image output size
- This ensures quadrants are calculated correctly

**Q: Can I have custom quadrant sizes?**
- No, this simplified version uses equal quadrants only
- For custom regions, you would need a different approach (ControlNet, etc.)

## Changelog
- **v2.0**: Simplified interface with fixed quadrants
  - Removed manual coordinate inputs
  - Automatic 4-quadrant division (equal sizes)
  - Fixed to combined mode (spatial + grounding tokens)
  - Renamed regions: Upper Left, Upper Right, Lower Left, Lower Right
  - Much easier to use!
- v1.1: Added prompt_mode parameter with three modes
- v1.0: Initial release with manual coordinates

## References
- [Qwen2-VL Paper](https://arxiv.org/html/2409.12191v1) - Technical details on visual grounding
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) - Model documentation
