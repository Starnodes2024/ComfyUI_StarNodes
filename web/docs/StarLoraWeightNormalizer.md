# ⭐ Star LoRA Weight Normalizer

## Overview
The **Star LoRA Weight Normalizer** is specifically designed for **distilled models** that work best with total LoRA weights ≤ 1.0. It automatically normalizes multiple LoRA weights so their sum equals a target value while maintaining their relative proportions.

## The Problem with Distilled Models
Distilled models (like Turbo, Lightning, Hyper, etc.) are optimized for speed and efficiency. However, they have a limitation:
- ✅ **Works well**: Single LoRA with weight ≤ 1.0
- ❌ **Bad results**: Multiple LoRAs with total weight > 1.0
- ❌ **Bad results**: Adding LoRAs without weight control

## The Solution
This node solves the problem by:
1. **Accepting** multiple LoRAs with your desired relative strengths
2. **Calculating** the total weight of all LoRAs
3. **Normalizing** all weights proportionally to reach your target total
4. **Maintaining** the relative balance between LoRAs

## Example

### Without Normalization (Bad for Distilled Models)
```
LoRA A: 0.8
LoRA B: 0.6
LoRA C: 0.4
Total: 1.8 ❌ (Too high for distilled models!)
```

### With Normalization (Perfect!)
```
Target Total Weight: 1.0

LoRA A: 0.8 → 0.44 (0.8/1.8 × 1.0)
LoRA B: 0.6 → 0.33 (0.6/1.8 × 1.0)
LoRA C: 0.4 → 0.22 (0.4/1.8 × 1.0)
Total: 1.0 ✅
```

Notice how the **relative proportions are preserved**:
- LoRA A is still the strongest
- LoRA B is medium
- LoRA C is the weakest
- But now they sum to exactly 1.0!

## Node Variants

### 1. Star LoRA Weight Normalizer (Full)
- Normalizes both **MODEL** and **CLIP** weights
- Separate normalization for each
- Optional CLIP input

### 2. Star LoRA Weight Normalizer (Model Only)
- Normalizes **MODEL** weights only
- Simpler, faster
- No CLIP input needed

## Inputs

### Required
- **model**: The diffusion model to apply LoRAs to
- **target_total_weight**: Target sum of all LoRA weights (default: 1.0)
  - For distilled models: Use **1.0 or less**
  - For standard models: Can use higher values if needed

### Optional (Full Version Only)
- **clip**: CLIP model to apply LoRAs to
- **normalize_model**: Enable/disable model weight normalization (default: True)
- **normalize_clip**: Enable/disable CLIP weight normalization (default: True)

### Dynamic LoRA Slots
- **lora1_name**: First LoRA (always visible)
- **strength1_model**: Model weight for first LoRA
- **strength1_clip**: CLIP weight for first LoRA (full version only)
- Add more LoRA slots dynamically from the UI!

## Outputs

### Full Version
1. **model**: Model with normalized LoRAs applied
2. **clip**: CLIP with normalized LoRAs applied
3. **normalization_info**: Detailed text showing:
   - Original weights
   - Normalization factors
   - Final normalized weights
   - Total weights before/after

### Model Only Version
1. **model**: Model with normalized LoRAs applied
2. **normalization_info**: Detailed normalization report

## Usage Tips

### For Distilled Models (Recommended)
```
Target Total Weight: 1.0 (or 0.8 for more subtle effects)
Normalize Model: True
Normalize CLIP: True

LoRA 1: Character LoRA (strength: 0.7)
LoRA 2: Style LoRA (strength: 0.5)
LoRA 3: Detail LoRA (strength: 0.3)

Result:
Character: 0.47 (strongest influence)
Style: 0.33 (medium influence)
Detail: 0.20 (subtle influence)
Total: 1.0 ✅
```

### For Standard Models
```
Target Total Weight: 2.0 (or higher if needed)
You can use higher weights with standard models
```

### Workflow Integration
1. Load your distilled model
2. Add Star LoRA Weight Normalizer node
3. Set target_total_weight to 1.0
4. Add your LoRAs with desired relative strengths
5. The node handles normalization automatically!

## Normalization Info Output
The node outputs detailed information to both:
- **Console**: Printed during execution
- **normalization_info output**: Can be displayed with ShowText node

Example output:
```
=== LoRA Weight Normalization ===
Target Total Weight: 1.0

Model Weights:
  Original Total: 1.8000
  Normalization Factor: 0.5556
  Normalized Total: 1.0000

CLIP Weights:
  Original Total: 1.5000
  Normalization Factor: 0.6667
  Normalized Total: 1.0000

Applied LoRAs:
  character_lora.safetensors:
    Model: 0.800 → 0.444
    CLIP: 0.700 → 0.467
  style_lora.safetensors:
    Model: 0.600 → 0.333
    CLIP: 0.500 → 0.333
  detail_lora.safetensors:
    Model: 0.400 → 0.222
    CLIP: 0.300 → 0.200
```

## When to Use This Node

### ✅ Perfect For:
- **Distilled models** (Turbo, Lightning, Hyper, etc.)
- **Multiple LoRAs** on distilled models
- **Consistent results** with predictable total weight
- **Experimenting** with LoRA combinations while staying within limits

### ⚠️ Not Needed For:
- Single LoRA usage
- Standard models where you want full control
- When total weight is already ≤ target

## Technical Details

### Normalization Formula
```
normalized_weight = original_weight × (target_total / sum_of_all_weights)
```

### Features
- **Proportional scaling**: Maintains relative strength ratios
- **Separate normalization**: Model and CLIP weights normalized independently
- **Dynamic slots**: Add as many LoRAs as needed
- **Caching**: LoRAs are cached for performance
- **Detailed reporting**: Full transparency on weight adjustments

## Category
`⭐StarNodes/Sampler`

## Related Nodes
- **Star 3 LoRAs**: Fixed 3 LoRA slots without normalization
- **Star Dynamic LoRA**: Dynamic LoRA slots without normalization
- **Star Dynamic LoRA (Model Only)**: Model-only without normalization

---

**Perfect for getting the best results from distilled models with multiple LoRAs! 🚀**
