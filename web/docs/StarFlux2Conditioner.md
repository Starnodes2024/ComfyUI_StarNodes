# ⭐ Star Flux2 Conditioner

An advanced conditioning node designed for Flux2 and similar diffusion models that combines text prompt encoding with optional reference image processing for enhanced image-to-image workflows.

**Category:** `⭐StarNodes/Conditioning`

## Overview

The Star Flux2 Conditioner streamlines the conditioning process by:
- Encoding text prompts using CLIP
- Processing up to 5 reference images as guiding latents
- Automatically resizing images to optimal dimensions (1 megapixel)
- Generating both positive and negative conditioning outputs
- Injecting reference latents into the conditioning structure for advanced control

This node is particularly useful for Flux2 edit models and other architectures that support reference-based generation.

## Inputs

### Required
- **clip (CLIP)**: The CLIP model used for text tokenization and encoding
- **vae (VAE)**: VAE model used to encode reference images into latent space
- **text (STRING)**: Your prompt text (supports multiline input)
- **join_references (BOOLEAN)**: When enabled (default: true), combines images 2-5 into a single grid before encoding. Image 1 is always processed separately

### Optional
- **image_1 (IMAGE)**: First optional reference image
- **image_2 (IMAGE)**: Second optional reference image
- **image_3 (IMAGE)**: Third optional reference image
- **image_4 (IMAGE)**: Fourth optional reference image
- **image_5 (IMAGE)**: Fifth optional reference image

## Outputs

- **POS (CONDITIONING)**: Positive conditioning with text embeddings and optional reference latents
- **NEG (CONDITIONING)**: Negative conditioning (empty prompt by default)
- **GRID_IMAGE (IMAGE)**: The created 2x2 grid image (when join_references is enabled and images 2-5 are provided). Returns a small white placeholder image if no grid was created

## How It Works

### Text Encoding
1. The input text is tokenized using the provided CLIP model
2. Tokens are encoded to generate conditioning embeddings and pooled output
3. Both positive (from your prompt) and negative (empty) conditioning are created

### Reference Image Processing
When reference images are connected:
1. **Image 1** is always processed separately (if provided)
2. **Images 2-5** are handled based on the `join_references` setting:
   - **Join References = True (default)**: Images 2-5 are combined into a 2x2 grid before encoding
     - Each cell is 1024x1024 pixels
     - Images are scaled to fit within cells while preserving aspect ratio
     - White padding is added to center images in their cells
     - Empty cells (when fewer than 4 images) are filled with white
     - The complete 2048x2048 grid is then resized to 1MP before encoding
   - **Join References = False**: Each image is processed individually
3. Each image (or grid) is automatically resized to approximately 1 megapixel
4. Dimensions are adjusted to be multiples of 16 (VAE-compatible)
5. High-quality bicubic interpolation is used for resizing
6. Images are encoded using the VAE (RGB channels only)
7. Resulting latents are injected as `reference_latents` in the conditioning

### Intelligent Resizing
The node includes a smart scaling algorithm that:
- Calculates the current pixel count of each image
- Determines the scale factor needed to reach 1MP target
- Rounds dimensions to the nearest multiple of 16
- Preserves aspect ratio while ensuring VAE compatibility

## Use Cases

### Basic Text-to-Image
Connect only CLIP, VAE, and provide a text prompt for standard text-to-image generation with positive and negative conditioning outputs.

### Image-Guided Generation
Add one or more reference images to guide the generation process. The model will use these as visual references while following your text prompt.

### Multi-Reference Workflows
Use multiple reference images (up to 5) to provide diverse visual guidance:
- **Join References = True**: Combines images 2-5 into a single grid, making it easier for models to process multiple references as one cohesive visual context. This is ideal when you have related reference images that should be viewed together.
- **Join References = False**: Processes each image separately, giving the model individual reference latents. Use this when each reference image represents a distinct concept or style.

### Edit and Refinement
Perfect for Flux2 edit models that support reference latents for tasks like style transfer, image variation, or guided editing.

### Grid Image Preview
The GRID_IMAGE output allows you to preview the 2x2 grid that was created from your reference images. This is useful for:
- Verifying the grid layout before generation
- Saving the grid for documentation
- Using the grid in other parts of your workflow

## Technical Details

- **Latent Batching**: Multiple reference images are concatenated along the batch dimension
- **VAE Encoding**: Only RGB channels (first 3) are used for encoding
- **Conditioning Structure**: Follows ComfyUI's standard conditioning format with `pooled_output` and optional `guiding_latent` keys
- **Memory Efficient**: Images are resized to 1MP before encoding to balance quality and memory usage

## Tips & Best Practices

1. **Reference Image Quality**: Higher quality reference images generally produce better results
2. **Consistent Styles**: Using reference images with similar styles can create more coherent outputs
3. **Prompt Alignment**: Ensure your text prompt aligns with the content of your reference images
4. **Model Compatibility**: This node works best with models specifically designed to use reference latents (like Flux2 edit models)
5. **Negative Conditioning**: The negative output uses an empty prompt by default, suitable for most Flux/SDXL workflows
6. **Join References Usage**:
   - **Enable (True)** when you have multiple related references (e.g., different angles of the same subject, color palette examples, or style references that work together)
   - **Disable (False)** when each reference represents a different concept or when the model struggles with grid layouts
   - Image 1 is always separate, so use it for your primary/most important reference

## Compatibility

- **Flux2 Models**: Fully compatible with Flux2 and Flux2 edit models
- **SDXL Models**: Works with SDXL and similar architectures
- **Reference-Aware Models**: Optimized for models that support `guiding_latent` or `reference_latents` conditioning

## Example Workflow

```
CLIP → Star Flux2 Conditioner → Sampler
VAE ↗         ↓ (reference images)
Text Prompt → 
```

## Changelog

- **v1.1 (1.9.9)**: Added `join_references` boolean option to combine images 2-5 into a grid layout before encoding, making it easier for models to handle multiple references
- **v1.0 (1.9.9)**: Initial release with support for 5 reference images, automatic resizing, and reference latent injection
