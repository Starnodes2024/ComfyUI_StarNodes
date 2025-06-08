# Star InfiniteYou Face Swap Mod

## Description
The Star InfiniteYou Face Swap Mod is an advanced face swapping node that allows you to transfer facial features from a reference image to generated images. It uses the InfiniteYou model to extract facial landmarks and embeddings, which are then applied during the diffusion process to guide the generation toward the target face. This node is particularly useful for creating consistent characters, celebrity likenesses, or personalized portraits.

## Inputs

### Required
- **control_net**: ControlNet model for applying facial landmarks
- **model**: Diffusion model to use for generation
- **image**: Source image containing the face to be processed
- **clip**: CLIP model for text encoding
- **adapter_file**: InfiniteYou adapter model file
- **start_at**: Starting point in the diffusion process (0.0-1.0)
- **end_at**: Ending point in the diffusion process (0.0-1.0)
- **vae**: VAE model for encoding/decoding images
- **weight**: Strength of the face swap effect (0.0-1.0)
- **blur_kernel**: Size of blur kernel for mask processing (odd numbers only)

### Optional
- **ref_image**: Reference image containing the target face (alternative to patch_file)
- **patch_file**: Pre-saved face patch to apply (alternative to ref_image)
- **cn_strength**: ControlNet strength multiplier (0.0-2.0)
- **noise**: Amount of noise to add to facial landmarks (0.0-1.0)
- **mask**: Optional mask to restrict face swap to specific areas
- **combine_embeds**: Method to combine embeddings ("average", "add", "subtract", "multiply")

## Outputs
- **MODEL**: The model with face swap settings applied
- **positive**: Positive conditioning with face embeddings
- **negative**: Negative conditioning with face embeddings
- **latent**: Latent representation for generation

## Usage
1. Connect a ControlNet model designed for face or pose control
2. Provide either:
   - A reference image containing the target face via the "ref_image" input
   - A pre-saved face patch via the "patch_file" dropdown
3. Adjust the strength, timing, and other parameters to control the face swap effect
4. Connect the outputs to a sampler node to generate images with the target face

## Features

### Face Detection and Processing
- Automatically detects and processes faces in reference images
- Extracts facial landmarks for precise feature positioning
- Creates face embeddings that capture the identity and characteristics

### Flexible Application Methods
- Apply faces directly from reference images
- Use pre-saved face patches for consistent results
- Random face selection option for variety

### Fine Control
- Adjustable strength for controlling how strongly the face is applied
- Start/end parameters to control when in the diffusion process the face is applied
- Noise parameter to add variation and prevent overfitting
- Optional masking for targeted application

### Embedding Combination Methods
- Average: Balanced combination of embeddings (default)
- Add: Emphasize combined features
- Subtract: Create contrast between features
- Multiply: Intensify common features

## Technical Details
- Uses the InfiniteYou model to extract facial features and embeddings
- Applies face information through ControlNet conditioning
- Processes both positive and negative conditioning for consistent results
- Supports various mask processing techniques for seamless integration
- Compatible with most diffusion models and samplers

## Notes
- For best results, use reference images with clear, front-facing faces
- The blur_kernel parameter affects how smoothly the face blends with the rest of the image
- Higher weight values produce stronger resemblance to the reference face
- The start_at and end_at parameters can be adjusted to balance identity preservation and image quality
- This node works best with models that have good face generation capabilities
