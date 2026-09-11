# SD(XL) Star(t) Settings

## Description
The SD(XL) Star(t) Settings node is a comprehensive starter node designed to initialize and configure Stable Diffusion XL workflows. It combines checkpoint loading, VAE selection, prompt conditioning, latent image generation, and aspect ratio management into a single node, streamlining the workflow setup process.

## Inputs

### Required
- **text**: Positive prompt text (multiline with dynamic prompt support)
- **negative_text**: Negative prompt text (multiline with dynamic prompt support)
- **Checkpoint**: Model checkpoint selection from available checkpoints
- **VAE**: VAE model selection (Default or custom)
- **Latent_Ratio**: Predefined aspect ratio selection (e.g., "1:1 [1024x1024 square]")
- **Latent_Width**: Custom width for latent image (used with "Free Ratio")
- **Latent_Height**: Custom height for latent image (used with "Free Ratio")
- **Batch_Size**: Number of images to generate in a batch

### Optional
- **LoRA_Stack**: Stack of LoRA models to apply to both the main model and conditioning

## Outputs
- **model**: Loaded checkpoint model with LoRAs applied (if provided)
- **clip**: CLIP model with LoRAs applied (if provided)
- **vae**: VAE model (default from checkpoint or custom)
- **latent**: Empty latent image with specified dimensions
- **width**: Output width in pixels
- **height**: Output height in pixels
- **conditioning_POS**: Positive conditioning from the text prompt
- **conditioning_NEG**: Negative conditioning from the negative text prompt

## Usage
1. Select a checkpoint model compatible with SDXL
2. Enter positive and negative prompts
3. Choose a predefined aspect ratio or use "Free Ratio" with custom dimensions
4. Optionally connect a LoRA stack for additional model customization
5. Connect the outputs to a sampler node (like SDstarsampler or KSampler)

## Features
- **Integrated Workflow Setup**: Combines multiple initialization steps into a single node
- **Aspect Ratio Management**: Includes predefined ratios optimized for SDXL
- **Custom Ratio Support**: Allows free ratio selection with custom dimensions
- **LoRA Integration**: Applies LoRAs to both the model and conditioning
- **Default Prompts**: Provides fallback prompts if none are specified
- **Dimension Correction**: Ensures dimensions are divisible by 8 for proper latent space processing

## Aspect Ratio System
- Predefined ratios are loaded from `sdratios.json` in the starnodes directory
- Custom user ratios can be defined in `user_ratios.json` in the ComfyUI base directory
- The "Free Ratio" option allows manual width and height specification

## Notes
- Despite the name, this node works with both SD and SDXL models
- When using the "Free Ratio" option, the specified width and height will be used directly
- All dimensions are automatically adjusted to be divisible by 8
- If no prompts are provided, default prompts will be used
- The node applies LoRAs to both the main model and a separate copy of the CLIP model used for conditioning
