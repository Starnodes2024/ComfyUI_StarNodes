# Star InfiniteYou Patch Combine

## Description
The Star InfiniteYou Patch Combine node allows you to blend multiple face patches together with weighted combinations, creating hybrid facial characteristics. This advanced node enables creative face mixing, style blending, and the creation of unique synthetic identities by combining up to five different face patches with precise control over their influence.

## Inputs

### Required
- **control_net**: ControlNet model for applying facial landmarks
- **model**: Diffusion model to use for generation
- **positive**: Positive conditioning to modify with face data
- **negative**: Negative conditioning to modify with face data
- **vae**: VAE model for encoding/decoding images
- **latent_image**: Latent representation for generation
- **patch_file_1**: First face patch to apply
- **weight_1**: Weight of the first face patch (0.0-1.0)
- **patch_file_2**: Second face patch to apply
- **weight_2**: Weight of the second face patch (0.0-1.0)

### Optional
- **patch_file_3**: Third face patch to apply
- **weight_3**: Weight of the third face patch (0.0-1.0)
- **patch_file_4**: Fourth face patch to apply
- **weight_4**: Weight of the fourth face patch (0.0-1.0)
- **patch_file_5**: Fifth face patch to apply
- **weight_5**: Weight of the fifth face patch (0.0-1.0)
- **cn_strength**: Overall ControlNet strength multiplier (0.0-2.0)
- **start_at**: Starting point in the diffusion process (0.0-1.0)
- **end_at**: Ending point in the diffusion process (0.0-1.0)

## Outputs
- **MODEL**: The model with combined face patch settings applied
- **positive**: Positive conditioning with combined face embeddings
- **negative**: Negative conditioning with combined face embeddings
- **latent**: Latent representation for generation
- **patch_data**: Combined patch data that can be saved with the Patch Saver node

## Usage
1. Connect a ControlNet model designed for face or pose control
2. Select at least two face patches to combine
3. Adjust the weights to control the influence of each face
4. Optionally add more face patches (up to five total)
5. Fine-tune the overall strength and timing parameters
6. Connect the outputs to a sampler node or save the combined patch

## Features

### Multi-Face Blending
- Combine up to five different face patches with weighted influence
- Create unique hybrid identities with characteristics from multiple faces
- Adjust individual weights to fine-tune the contribution of each face

### Intelligent Dimension Handling
- Automatically handles dimension mismatches between face patches
- Resizes facial landmarks to ensure compatibility
- Maintains proper alignment of facial features

### Weighted Combination
- Precise control over the influence of each face patch
- Weights are normalized to ensure consistent results
- Fine-tune the balance between different facial characteristics

### Comprehensive Parameter Control
- Adjust overall ControlNet strength for the combined face
- Control when in the diffusion process the face is applied
- Create reusable combined patches by connecting to a Patch Saver

## Technical Details
- Loads and processes multiple face patches simultaneously
- Performs weighted averaging of facial landmarks and embeddings
- Handles dimension mismatches using bilinear interpolation
- Applies combined face data through ControlNet conditioning
- Processes both positive and negative conditioning for consistent results
- Outputs combined patch data that can be saved for future use

## Notes
- Face patches must be created first using the Star InfiniteYou Patch Saver node
- For best results, use patches created from clear, front-facing reference images
- Experiment with different weight combinations to achieve desired facial characteristics
- The combined patch can be saved using the Star InfiniteYou Patch Saver node
- This node works best with models that have good face generation capabilities
