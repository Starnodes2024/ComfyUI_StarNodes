# Star InfiniteYou Patch Loader

## Description
The Star InfiniteYou Patch Loader is a specialized node that loads pre-saved face data (patches) and applies them to diffusion models during the generation process. This node allows you to consistently apply specific faces across multiple workflows without needing the original reference images. It works in conjunction with other InfiniteYou nodes to provide a complete face-swapping and identity preservation system.

## Inputs

### Required
- **control_net**: ControlNet model for applying facial landmarks
- **model**: Diffusion model to use for generation
- **positive**: Positive conditioning to modify with face data
- **negative**: Negative conditioning to modify with face data
- **vae**: VAE model for encoding/decoding images
- **latent_image**: Latent representation for generation
- **patch_file**: Selection of available face patches or special options:
  - "none": Skip face application
  - "random": Randomly select a face patch for each batch item
  - [List of saved patches]: Specific face patch to apply

## Outputs
- **MODEL**: The model with face patch settings applied
- **positive**: Positive conditioning with face embeddings
- **negative**: Negative conditioning with face embeddings
- **latent**: Latent representation for generation

## Usage
1. Connect a ControlNet model designed for face or pose control
2. Select a previously saved face patch from the dropdown
3. Connect the outputs to a sampler node to generate images with the target face
4. For random face selection, choose the "random" option

## Features

### Patch Management
- Automatically detects and lists available face patches
- Supports both .pt and .iyou file formats for compatibility
- Alphabetically sorts patches for easier selection
- Provides "none" and "random" options for flexibility

### Consistent Face Application
- Applies the same face data consistently across generations
- Preserves facial identity characteristics from the original reference
- Maintains facial landmark positioning for accurate feature placement

### Random Face Selection
- Optional random mode selects a different face for each batch item
- Great for creating variety while maintaining quality
- Uses all available face patches in the patch directory

### Batch Processing
- Supports processing multiple images in a batch
- Applies face data consistently across all batch items when using a specific patch
- Can apply different random faces to each batch item when using random mode

## Technical Details
- Loads face patches from the "infiniteyoupatch" directory in the ComfyUI output folder
- Each patch contains facial landmarks, embeddings, and control parameters
- Applies face data through ControlNet conditioning
- Processes both positive and negative conditioning for consistent results
- Compatible with most diffusion models and samplers

## Notes
- Face patches must be created first using the Star InfiniteYou Patch Saver node
- For best results, use patches created from clear, front-facing reference images
- The patch contains all necessary parameters including strength and timing settings
- This node works best with models that have good face generation capabilities
- The "random" option requires at least one saved face patch to function
