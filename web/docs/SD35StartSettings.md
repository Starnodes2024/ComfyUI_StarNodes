# SD3.5 Star(t) Settings

## Description
The SD3.5 Star(t) Settings node is a specialized starter node designed specifically for Stable Diffusion 3.5 models in ComfyUI. It provides comprehensive initialization for SD3.5 workflows, handling model loading, device management, VAE selection, prompt conditioning, and latent image generation in a single node. This node is optimized for the unique requirements of SD3.5 models.

## Inputs

### Required
- **text**: Positive prompt text (multiline with dynamic prompt support)
- **negative_text**: Negative prompt text (multiline with dynamic prompt support)
- **UNET**: UNET model selection (Default or custom)
- **CLIP_1**: Primary CLIP model selection (Default or custom)
- **CLIP_2**: Secondary CLIP model selection (Default or custom)
- **CLIP_3**: Tertiary CLIP model selection (Default or custom)
- **CLIP_Device**: Device for CLIP models (CPU or CUDA)
- **VAE**: VAE model selection (Default, custom, or TAESD variants)
- **VAE_Device**: Device for VAE processing (CPU or CUDA)
- **Weight_Dtype**: Model weight data type (float32, float16, bfloat16)
- **Latent_Ratio**: Predefined aspect ratio selection
- **Latent_Width**: Custom width for latent image (used with "Free Ratio")
- **Latent_Height**: Custom height for latent image (used with "Free Ratio")
- **Batch_Size**: Number of images to generate in a batch

### Optional
- **LoRA_Stack**: Stack of LoRA models to apply to both the main model and conditioning

## Outputs
- **model**: Loaded UNET model with LoRAs applied (if provided)
- **clip**: CLIP model with LoRAs applied (if provided)
- **latent**: Empty latent image with specified dimensions
- **width**: Output width in pixels
- **height**: Output height in pixels
- **conditioning_pos**: Positive conditioning from the text prompt
- **conditioning_neg**: Negative conditioning from the negative text prompt
- **vae**: VAE model (default from checkpoint or custom)

## Usage
1. Select UNET and CLIP models compatible with SD3.5
2. Enter positive and negative prompts
3. Choose appropriate devices for CLIP and VAE processing
4. Select a VAE (including special TAESD options for different model types)
5. Choose a predefined aspect ratio or use "Free Ratio" with custom dimensions
6. Optionally connect a LoRA stack for additional model customization
7. Connect the outputs to a compatible sampler node

## Features
- **SD3.5 Optimization**: Specifically configured for Stable Diffusion 3.5 models
- **Multi-CLIP Support**: Handles up to three CLIP models for SD3.5's architecture
- **Advanced Device Management**: Separate device selection for CLIP and VAE models
- **TAESD Support**: Built-in support for TAESD, TAESDXL, TAESD3, and TAEF1 approximate VAEs
- **Weight Precision Control**: Options for float32, float16, and bfloat16 precision
- **Aspect Ratio Management**: Includes predefined ratios optimized for various models
- **Custom Ratio Support**: Allows free ratio selection with custom dimensions
- **LoRA Integration**: Applies LoRAs to both the model and conditioning
- **Dimension Correction**: Ensures dimensions are divisible by 8 for proper latent space processing

## Aspect Ratio System
- Predefined ratios are loaded from a JSON file in the starnodes directory
- Custom user ratios can be defined in the ComfyUI base directory
- The "Free Ratio" option allows manual width and height specification

## Advanced Features
- **Device Override**: Automatically overrides model device settings for optimal performance
- **Automatic TAESD Detection**: Detects and makes available various TAESD VAE variants
- **SD3.5-Specific Conditioning**: Properly processes text into conditioning suitable for SD3.5 models

## Notes
- This node is specifically optimized for SD3.5 models and may not work correctly with other architectures
- When using the "Free Ratio" option, the specified width and height will be used directly
- All dimensions are automatically adjusted to be divisible by 8
- For optimal performance with SD3.5 models, use appropriate CLIP models and VAE
- The node applies LoRAs to both the main model and the CLIP model used for conditioning
