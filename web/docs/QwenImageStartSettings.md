# Qwen Image Star(t) Settings

## Description
The Qwen Image Star(t) Settings node is a specialized all-in-one starter node designed for Qwen diffusion models in ComfyUI. It provides comprehensive initialization for Qwen Image workflows, handling model loading, CLIP configuration, VAE selection, prompt conditioning (both positive and negative), and latent image generation with Qwen-optimized aspect ratios in a single node.

## Inputs

### Required

- **Positive_Prompt**: Multiline text input for positive prompt with dynamic prompt support
  - If left empty, defaults to: "a confused looking fluffy purple monster with a \"?\" sign"
  - Supports dynamic prompts and wildcards

- **Negative_Prompt**: Multiline text input for negative prompt with dynamic prompt support
  - If left empty, creates an empty zero-out condition for negative conditioning
  - Supports dynamic prompts and wildcards

- **Diffusion_Model**: Selector for diffusion models
  - Shows models from `models/diffusion_models` folder
  - Shows models from `models/unet` folder
  - Default option available for checkpoint-based loading

- **VAE**: VAE model selector
  - Shows models from `models/vae` folder
  - Includes TAESD variants (taesd, taesdxl, taesd3, taef1)
  - Default option available

- **CLIP**: CLIP model selector
  - Shows models from `models/clip` folder
  - Default option available

- **CLIP_Type**: CLIP model type selector
  - Options: stable_diffusion, stable_cascade, sd3, stable_audio, mochi, ltxv, flux, hunyuan_video, pixart, cosmos, lumina2, wan, hidream, chroma, ace, omnigen2, qwen_image, hunyuan_image
  - Default: qwen_image (optimized for Qwen models)
  - Determines how CLIP model is loaded and processed
  - For Qwen Image models, use "qwen_image"
  - For WAN models, use "wan"

- **CLIP_Device**: Device selection for CLIP model
  - Options: cpu, cuda:0, cuda:1, etc.
  - Default: cpu
  - Allows offloading CLIP to CPU to save VRAM

- **Latent_Ratio**: Predefined aspect ratios optimized for Qwen Image models
  - 1:1 (1328x1328) - Square format
  - 16:9 (1664x928) - Widescreen landscape
  - 9:16 (928x1664) - Vertical/portrait
  - 4:3 (1472x1104) - Standard landscape
  - 3:4 (1104x1472) - Standard portrait
  - 3:2 (1584x1056) - Classic photo landscape
  - 2:3 (1056x1584) - Classic photo portrait
  - 5:7 (1120x1568) - Poster portrait
  - 7:5 (1568x1120) - Poster landscape
  - Free Ratio (custom) - Use custom width/height

- **Latent_Width**: Custom width in pixels (default: 1328)
  - Range: 16 to 8192
  - Step: 16
  - Used when "Free Ratio (custom)" is selected

- **Latent_Height**: Custom height in pixels (default: 1328)
  - Range: 16 to 8192
  - Step: 16
  - Used when "Free Ratio (custom)" is selected

- **Batch_Size**: Number of images to generate in batch (default: 1)
  - Range: 1 to 4096
  - Generates multiple latents simultaneously

- **use_nearest_image_ratio**: Boolean toggle (default: False)
  - When enabled with an image input, automatically selects the closest matching aspect ratio
  - Overrides manual ratio selection when active

### Optional

- **image**: Optional image input
  - Used with `use_nearest_image_ratio` for automatic aspect ratio detection
  - Node analyzes input image dimensions and selects the nearest predefined ratio

## Outputs

1. **model**: Loaded diffusion model (MODEL type)
   - Ready for sampling operations
   - Can be None if "Default" is selected

2. **clip**: Loaded CLIP model (CLIP type)
   - Configured with specified device and type
   - Can be None if "Default" is selected

3. **vae**: Loaded VAE model (VAE type)
   - Ready for encoding/decoding operations
   - Can be None if "Default" is selected

4. **latent**: Empty latent image (LATENT type)
   - Dimensions based on selected ratio or custom size
   - Batch size applied

5. **width**: Output width in pixels (INT)
   - Actual width used for latent generation
   - Always divisible by 8

6. **height**: Output height in pixels (INT)
   - Actual height used for latent generation
   - Always divisible by 8

7. **condition_pos**: Positive conditioning (CONDITIONING type)
   - Generated from positive prompt using CLIP
   - Ready for sampler input

8. **condition_neg**: Negative conditioning (CONDITIONING type)
   - Generated from negative prompt using CLIP
   - Empty zero-out condition if negative prompt is empty

9. **prompt_pos**: Positive prompt text (STRING)
   - Original or default positive prompt
   - Useful for saving metadata or further processing

10. **prompt_neg**: Negative prompt text (STRING)
    - Original negative prompt text
    - Useful for saving metadata or further processing

## Usage

### Basic Workflow
1. Select your Qwen diffusion model from the Diffusion_Model dropdown
2. Choose an appropriate VAE model (or use Default)
3. Select a CLIP model and configure its type and device
4. Enter your positive prompt (or leave empty for default)
5. Optionally enter a negative prompt (empty creates zero-out condition)
6. Choose a predefined aspect ratio or use Free Ratio with custom dimensions
7. Set batch size if generating multiple images
8. Connect outputs to a sampler node

### Advanced Usage with Image Input
1. Enable `use_nearest_image_ratio`
2. Connect an image to the optional image input
3. Node will automatically select the closest matching Qwen ratio
4. Useful for img2img workflows or maintaining consistent aspect ratios

## Features

- **Qwen-Optimized Ratios**: Predefined aspect ratios specifically designed for Qwen Image models
- **Flexible Model Loading**: Supports models from both diffusion_models and unet folders
- **CLIP Type Selection**: Full control over CLIP model type for compatibility
- **Device Management**: Separate device selection for CLIP to optimize VRAM usage
- **TAESD Support**: Built-in support for TAESD approximate VAEs for faster previews
- **Automatic Ratio Detection**: Optional image-based aspect ratio selection
- **Default Prompt System**: Creative default prompt when left empty
- **Zero-Out Negative**: Automatic empty conditioning for negative when not specified
- **Dimension Correction**: Ensures all dimensions are properly divisible by 8 and 16
- **String Outputs**: Provides prompt strings for metadata and further processing

## Aspect Ratio System

The node uses Qwen-specific aspect ratios that are optimized for Qwen Image diffusion models:
- All ratios are pre-calculated for optimal quality
- Dimensions are always divisible by 16 for proper processing
- Free Ratio option allows complete customization
- Automatic nearest ratio detection available with image input

## CLIP Type Selection

The node supports all CLIP types available in ComfyUI. Select the appropriate type for your model:
- **qwen_image**: For Qwen Image diffusion models (recommended default)
- **wan**: For WAN (Wuerstchen Autoencoder Network) models
- **flux**: For FLUX models
- **sd3**: For Stable Diffusion 3 models
- **stable_diffusion**: For SD 1.x/2.x models
- **pixart**: For PixArt models
- **cosmos**: For Cosmos models
- **lumina2**: For Lumina 2 models
- **hidream**: For HiDream models
- **chroma**: For Chroma models
- **ace**: For ACE models
- **omnigen2**: For OmniGen2 models
- **hunyuan_image**: For Hunyuan Image models
- **hunyuan_video**: For Hunyuan Video models
- **mochi**: For Mochi models
- **ltxv**: For LTXV models
- **stable_cascade**: For Stable Cascade models
- **stable_audio**: For Stable Audio models

## Device Management

The node allows CLIP to be loaded on a different device (typically CPU) to save VRAM:
- Diffusion model typically stays on GPU for performance
- CLIP can be offloaded to CPU to reduce memory usage
- VAE follows standard ComfyUI device management

## Notes

- This node is specifically optimized for Qwen Image diffusion models
- When using "Free Ratio (custom)", ensure dimensions are reasonable for your model
- All dimensions are automatically adjusted to be divisible by 8 for latent space compatibility
- Empty negative prompt creates a proper zero-out condition, not just an empty string
- The default positive prompt generates a creative test image when no prompt is provided
- CLIP type should match your model architecture for proper conditioning
- String outputs (prompt_pos, prompt_neg) are useful for saving generation metadata

## Compatibility

- Works with Qwen Image diffusion models
- Compatible with standard ComfyUI samplers
- Supports all standard VAE models
- Works with various CLIP architectures through type selection
- Can be used with LoRA loaders and other model modification nodes

## Tips

- For VRAM-constrained systems, load CLIP on CPU
- Use predefined ratios for best results with Qwen models
- Enable `use_nearest_image_ratio` for img2img workflows
- Leave negative prompt empty for unconditional generation
- Use batch size > 1 for generating variations efficiently
- Connect string outputs to save nodes for metadata preservation
