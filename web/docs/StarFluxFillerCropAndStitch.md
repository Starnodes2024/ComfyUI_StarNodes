# Star FluxFill Inpainter

## Description
The Star FluxFill Inpainter is an advanced inpainting node specifically optimized for Flux models. It features intelligent crop-and-stitch functionality that automatically detects and processes only the masked regions, significantly improving memory efficiency and generation quality. This node combines differential attention techniques with optimized batch processing to create seamless inpainting results, particularly effective for large images and complex edits.

## Inputs

### Required
- **model**: The diffusion model to use (works best with Flux models)
- **text**: Prompt describing what should be inpainted in the masked area
- **vae**: VAE model for encoding/decoding images
- **image**: The source image to be inpainted
- **mask**: The mask indicating areas to be inpainted (white areas will be inpainted)
- **seed**: Random seed for reproducible results
- **steps**: Number of sampling steps
- **cfg**: Classifier-free guidance scale (how closely to follow the prompt)
- **sampler_name**: Sampling algorithm (euler, euler_ancestral, etc.)
- **scheduler**: Noise schedule (normal, karras, exponential, etc.)
- **denoise**: Denoising strength (1.0 for full inpainting, lower for subtle changes)
- **noise_mask**: Whether to apply noise only to the masked region
- **batch_size**: Number of samples to process in parallel (1-16)
- **differential_attention**: Whether to use differential attention for better boundary handling

### Optional
- **clip**: CLIP model (if not provided, will use the one from the model)
- **condition**: Pre-computed conditioning (if not provided, will generate from text)

## Outputs
- **image**: The inpainted result image
- **latent**: Latent representation of the inpainted image
- **mask**: The original mask (passed through)
- **clip**: The CLIP model used (passed through)
- **vae**: The VAE model used (passed through)
- **seed**: The seed used for generation

## Usage
1. Connect a Flux model to the model input
2. Provide an image to be inpainted
3. Create or connect a mask where white areas will be inpainted
4. Enter a text prompt describing what should appear in the masked area
5. Adjust sampling parameters as needed
6. Run the node to generate the inpainted result

## Features

### Smart Crop and Stitch
- Automatically detects and crops only the masked region with appropriate padding
- Processes only the necessary portion of the image, saving memory and computation
- Intelligently stitches the inpainted region back into the original image
- Handles multiple disconnected mask regions efficiently

### Differential Attention
- Uses a specialized attention mechanism that respects mask boundaries
- Improves coherence between inpainted regions and the original image
- Reduces artifacts and improves detail preservation at mask edges

### Batch Processing
- Can generate multiple variations in parallel for better GPU utilization
- Each batch item uses a different seed (base seed + batch index)
- Results are combined for easy comparison

### Memory Optimization
- Efficient tensor management to minimize VRAM usage
- Automatic cleanup of unused tensors
- Optimized for processing large images that would otherwise cause out-of-memory errors

## Technical Details
- The node uses a two-stage process: crop-and-prepare followed by inpaint-and-stitch
- Mask preprocessing includes optional growth and blurring for better boundary handling
- The inpainting process uses InpaintModelConditioning for proper noise conditioning
- Differential diffusion modifies the model's denoise function to respect mask boundaries
- The node is particularly optimized for Flux models but works with standard SD models as well

## Notes
- For best results with large images, use batch_size=1 and differential_attention=True
- When inpainting small regions in very large images, this node is significantly more efficient than standard inpainting
- The noise_mask option is usually beneficial but can be disabled for certain models if results are unsatisfactory
- This node works particularly well with Flux models that have strong inpainting capabilities
- For complex inpainting tasks, increasing steps to 30-50 often produces better results
