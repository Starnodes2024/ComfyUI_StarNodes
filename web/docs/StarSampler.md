# ⭐ StarSampler (Unified)

## Overview

**StarSampler** is a unified, all-in-one sampling node for ComfyUI that works seamlessly with both **Flux/AuraFlow** models and **SD/SDXL/SD3.5** models. It combines the functionality of the original FluxStarSampler and SDStarSampler into a single, intelligent node that automatically detects the model type and adjusts its behavior accordingly.

## Key Features

- **🔄 Automatic Model Detection**: Automatically detects whether you're using a Flux/AuraFlow model or an SD-based model
- **🎯 Optional Negative Conditioning**: Negative conditioning is optional - not needed for Flux models but available for SD models
- **✨ Detail Daemon Support**: Optional detail schedule for enhanced image quality
- **🔧 Updated VAE API**: Uses the latest ComfyUI VAE decode API for compatibility
- **🚀 Standalone**: Works independently without requiring comfyui_starnodes installation
- **⚡ Optimized Sampling**: Includes fast path for standard sampling and enhanced path for detail daemon

## Inputs

### Required Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **model** | MODEL | - | The diffusion model to use for sampling |
| **positive** | CONDITIONING | - | Positive conditioning (your prompt) |
| **latent** | LATENT | - | The latent image to denoise |
| **seed** | INT | 0 | Random seed for reproducible results |
| **steps** | INT | 20 | Number of sampling steps (1-10000) |
| **cfg** | FLOAT | 7.0 | Classifier Free Guidance scale (0.0-100.0) |
| **sampler_name** | SAMPLER | euler | Sampling algorithm to use |
| **scheduler** | SCHEDULER | beta | Noise schedule type |
| **denoise** | FLOAT | 1.0 | Denoising strength (0.0-1.0) |
| **vae** | VAE | - | VAE model for decoding latents to images |
| **decode_image** | BOOLEAN | True | Whether to decode latent to image |

### Optional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **negative** | CONDITIONING | None | Negative conditioning (optional, ignored for Flux) |
| **max_shift** | FLOAT | 1.15 | Max shift parameter for Flux models (ignored for SD) |
| **base_shift** | FLOAT | 0.5 | Base shift parameter for Flux/AuraFlow models |
| **detail_schedule** | DETAIL_SCHEDULE | None | Optional detail daemon schedule for enhanced quality |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| **model** | MODEL | The model (passed through) |
| **positive** | CONDITIONING | Positive conditioning (passed through) |
| **negative** | CONDITIONING | Negative conditioning (passed through or created) |
| **latent** | LATENT | The sampled latent image |
| **image** | IMAGE | Decoded image (if decode_image is True) |
| **vae** | VAE | The VAE (passed through) |
| **seed** | INT | The seed used (passed through) |

## Usage Examples

### Basic Usage with Flux Model

```
1. Load your Flux model with a checkpoint loader
2. Create positive conditioning with CLIP Text Encode
3. Create an empty latent or load an image
4. Connect everything to StarSampler
5. No need to connect negative conditioning for Flux!
```

**Typical Settings for Flux:**
- Steps: 20-30
- CFG: 3.5-5.0 (used as guidance)
- Sampler: euler
- Scheduler: beta
- Max Shift: 1.15
- Base Shift: 0.5

### Basic Usage with SD/SDXL/SD3.5 Model

```
1. Load your SD/SDXL/SD3.5 model with a checkpoint loader
2. Create positive conditioning with CLIP Text Encode
3. Create negative conditioning with CLIP Text Encode (optional but recommended)
4. Create an empty latent or load an image
5. Connect everything to StarSampler
```

**Typical Settings for SD/SDXL:**
- Steps: 20-40
- CFG: 7.0-9.0
- Sampler: euler, dpmpp_2m, etc.
- Scheduler: karras, normal, etc.
- Denoise: 1.0 for txt2img, 0.5-0.8 for img2img

### Advanced Usage with Detail Daemon

The Detail Daemon feature enhances image quality by adjusting the sampling process at specific steps. To use it:

1. Create a Detail Schedule node (if available in your setup)
2. Configure the detail parameters:
   - **detail_amount**: Strength of the detail enhancement (0.0-1.0)
   - **detail_start**: When to start applying detail (0.0-1.0)
   - **detail_end**: When to stop applying detail (0.0-1.0)
   - **detail_bias**: Bias towards start or end (0.0-1.0)
   - **detail_exponent**: Curve of the detail application (1.0-3.0)
3. Connect the detail schedule to the StarSampler

**Recommended Detail Settings:**
- Amount: 0.3-0.6
- Start: 0.2-0.3
- End: 0.7-0.9
- Bias: 0.5
- Exponent: 1.5-2.0

## Model-Specific Behavior

### Flux/AuraFlow Models

When StarSampler detects a Flux or AuraFlow model:
- **Negative conditioning is ignored** (Flux doesn't use negative prompts)
- **CFG is used as guidance** value in the conditioning
- **Max shift and base shift** parameters are applied for model sampling
- Uses advanced custom sampler with guider system
- Optimized for Flux's flow-matching architecture

### SD/SDXL/SD3.5 Models

When StarSampler detects an SD-based model:
- **Negative conditioning is used** (creates empty if not provided)
- **CFG works as traditional classifier-free guidance**
- **Max shift and base shift** are ignored
- Uses standard KSampler approach
- Compatible with all SD variants

## Technical Details

### VAE Decoding

StarSampler uses the updated ComfyUI VAE API:
```python
image = vae.decode(latent["samples"])
```

This ensures compatibility with the latest ComfyUI versions where the VAE encode/decode API has changed.

### Detail Daemon Algorithm

The Detail Daemon creates a schedule that modifies sigma values during sampling:
1. Creates a bell-curve or custom-curve schedule based on parameters
2. Interpolates between schedule points based on current sigma
3. Adjusts sigma values to enhance detail at specific steps
4. Applies adjustments through forward method wrapping

### Performance Optimization

- **Fast Path**: When no detail schedule is connected, uses optimized sampling path
- **Detail Path**: When detail schedule is connected, uses enhanced sampling with sigma adjustments
- **Memory Efficient**: Clones latents only when necessary
- **Device Aware**: Automatically handles device placement for tensors

## Troubleshooting

### Issue: "No negative conditioning provided for SD model"

**Solution**: This is just a warning. The node will create an empty negative conditioning automatically. For better results with SD models, connect a negative prompt.

### Issue: Image not decoding

**Solution**: Make sure `decode_image` is set to True and a valid VAE is connected.

### Issue: Flux model producing poor results

**Solution**: 
- Try adjusting CFG (guidance) to 3.5-5.0 range
- Ensure max_shift is around 1.15 and base_shift around 0.5
- Use 20-30 steps with euler sampler and beta scheduler

### Issue: SD model ignoring negative prompt

**Solution**: Make sure you're connecting the negative conditioning to the optional `negative` input.

## Compatibility

- ✅ ComfyUI (latest version)
- ✅ Flux models (all variants)
- ✅ AuraFlow models
- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion XL
- ✅ Stable Diffusion 3.5
- ✅ Works standalone without comfyui_starnodes

## Tips & Best Practices

1. **Start Simple**: Begin with default settings and adjust from there
2. **Model-Specific Tuning**: Different models need different CFG values
3. **Step Count**: More steps = better quality but slower generation
4. **Detail Daemon**: Use sparingly, not needed for all images
5. **Seed Control**: Use fixed seeds for reproducible results
6. **Batch Processing**: The node supports batch latents
7. **Memory Management**: Higher resolutions need more VRAM

## Credits

- Detail Daemon adapted from [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon)
- Detail Daemon adapted from [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
- Original StarSampler concepts from comfyui_starnodes
- Unified implementation for ComfyUI_StarBetaNodes

## Version History

- **v1.0.0** (2024): Initial unified release
  - Combined Flux and SD samplers into one node
  - Optional negative conditioning
  - Updated VAE API support
  - Standalone functionality

---

**Category**: ⭐StarBetaNodes/Sampler

**Node Name**: StarSampler

**Display Name**: ⭐ StarSampler (Unified)
