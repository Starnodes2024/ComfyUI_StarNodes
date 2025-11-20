# Star Qwen Regional Prompter v2.0 - Simplified!

## What Changed

Based on your feedback that **combined mode works best** and to make the node **much easier to use**, I've completely simplified the interface!

## v2.0 Changes

### ✅ What's New

1. **Fixed 4-Quadrant Layout**
   - No more manual coordinates!
   - Image automatically divided into 4 equal parts
   - Upper Left, Upper Right, Lower Left, Lower Right

2. **Combined Mode Only**
   - Removed mode selection
   - Always uses combined mode (spatial descriptions + grounding tokens)
   - This is what tested best for Qwen-Image!

3. **Simpler Inputs**
   - Just 4 optional text fields for quadrants
   - No X, Y, Width, Height inputs
   - Much cleaner interface!

4. **Removed Debug Mode**
   - No longer needed with simplified interface
   - Node is straightforward now

### ❌ What's Removed

- Manual coordinate inputs (X, Y, Width, Height)
- Mode selection (spatial_description, grounding_tokens, combined)
- Debug print option
- All the complexity!

## New Interface

### Required Inputs
- `clip` - Qwen2.5-VL CLIP model
- `background_prompt` - Overall scene description
- `image_width` - Target image width (e.g., 1024)
- `image_height` - Target image height (e.g., 1024)

### Optional Inputs (Quadrants)
- `region_upper_left` - Prompt for upper left quadrant
- `region_upper_right` - Prompt for upper right quadrant
- `region_lower_left` - Prompt for lower left quadrant
- `region_lower_right` - Prompt for lower right quadrant

**Leave empty to skip that quadrant!**

## How It Works Now

### Automatic Division

For a 1024x1024 image:

```
┌──────────┬──────────┐
│          │          │
│  Upper   │  Upper   │
│  Left    │  Right   │
│  (512x   │  (512x   │
│   512)   │   512)   │
├──────────┼──────────┤
│          │          │
│  Lower   │  Lower   │
│  Left    │  Right   │
│  (512x   │  (512x   │
│   512)   │   512)   │
└──────────┴──────────┘
```

### Example Usage

**Simple Top/Bottom Split:**
```
Background: "Beautiful landscape, golden hour"
Image: 1024x1024

Region Upper Left: "dramatic mountain peaks"
Region Upper Right: "dramatic mountain peaks"
Region Lower Left: "serene alpine lake"
Region Lower Right: "serene alpine lake"
```

**Four Distinct Quadrants:**
```
Background: "Fantasy scene, magical atmosphere"
Image: 1024x1024

Region Upper Left: "fire element, flames"
Region Upper Right: "water element, waves"
Region Lower Left: "earth element, rocks"
Region Lower Right: "air element, clouds"
```

**Partial Control (only one quadrant):**
```
Background: "Professional portrait, studio lighting"
Image: 1024x1024

Region Upper Left: "detailed face with expressive eyes"
Region Upper Right: (empty)
Region Lower Left: (empty)
Region Lower Right: (empty)
```

## Benefits of v2.0

### 🎯 Much Easier to Use
- No coordinate math needed
- No confusion about X, Y, Width, Height
- Just think in quadrants: Upper Left, Upper Right, Lower Left, Lower Right

### ⚡ Faster Workflow
- Fewer inputs to configure
- No mode selection needed
- Just fill in the quadrants you want

### 🔄 Optimized for Qwen-Image
- Uses combined mode which you confirmed works best
- Automatic coordinate calculation
- No manual tuning needed

### 🧩 More Intuitive
- Think spatially: "I want mountains on top, lake on bottom"
- Fill both upper quadrants with "mountains"
- Fill both lower quadrants with "lake"
- Done!

## Migration from v1.0

If you were using the old version with manual coordinates:

**Old way:**
```
Region 1:
  Prompt: "mountains"
  X: 0, Y: 0
  Width: 1024, Height: 512
```

**New way:**
```
Region Upper Left: "mountains"
Region Upper Right: "mountains"
```

Much simpler!

## Common Patterns

### Top Half Control
```
Region Upper Left: "your prompt"
Region Upper Right: "your prompt"
Region Lower Left: (empty)
Region Lower Right: (empty)
```

### Bottom Half Control
```
Region Upper Left: (empty)
Region Upper Right: (empty)
Region Lower Left: "your prompt"
Region Lower Right: "your prompt"
```

### Left Half Control
```
Region Upper Left: "your prompt"
Region Upper Right: (empty)
Region Lower Left: "your prompt"
Region Lower Right: (empty)
```

### Right Half Control
```
Region Upper Left: (empty)
Region Upper Right: "your prompt"
Region Lower Left: (empty)
Region Lower Right: "your prompt"
```

### All Four Quadrants
```
Region Upper Left: "prompt A"
Region Upper Right: "prompt B"
Region Lower Left: "prompt C"
Region Lower Right: "prompt D"
```

## Technical Details

### Quadrant Calculation
```python
half_width = image_width // 2
half_height = image_height // 2

# Upper Left: (0, 0) to (half_width, half_height)
# Upper Right: (half_width, 0) to (image_width, half_height)
# Lower Left: (0, half_height) to (half_width, image_height)
# Lower Right: (half_width, half_height) to (image_width, image_height)
```

### Combined Mode Format
Each quadrant with a prompt is formatted as:
```
[prompt] in the [position] area <|object_ref_start|>[prompt]<|object_ref_end|> <|box_start|>(x1,y1),(x2,y2)<|box_end|>
```

Example:
```
Input: region_upper_left = "mountains"
Output: "mountains in the upper left area <|object_ref_start|>mountains<|object_ref_end|> <|box_start|>(0,0),(499,499)<|box_end|>"
```

## Files Updated

1. **star_qwen_regional_prompter.py** - Completely rewritten for quadrant interface
2. **web/docs/StarQwenRegionalPrompter.md** - Updated documentation
3. **README_StarQwenRegionalPrompter.md** - Updated README

## Next Steps

1. **Restart ComfyUI** to load the updated node
2. **Try the new interface** - it's much simpler!
3. **Test with your workflows** - combined mode should work great
4. **Enjoy easier regional prompting!**

## Feedback Welcome

This is v2.0 based on your feedback that combined mode works best. If you have any suggestions for further improvements, let me know!

---

**Version**: 2.0  
**Date**: 2025-01-11  
**Status**: Production Ready  
**Tested With**: Qwen2.5-VL CLIP + Qwen-Image (MMDiT)
