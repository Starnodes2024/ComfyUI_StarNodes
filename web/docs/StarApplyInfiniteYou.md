# Star InfiniteYou Apply

## Description
The Star InfiniteYou Apply node is the primary node for extracting and applying facial features from a reference image to generated images. It uses the InfiniteYou model to analyze a reference face, extract its facial landmarks and embeddings, and apply them during the diffusion process. This node is ideal for creating consistent characters, celebrity likenesses, or personalized portraits directly from reference images.

## Inputs

### Required
- **control_net**: ControlNet model for applying facial landmarks
- **model**: Diffusion model to use for generation
- **positive**: Positive conditioning to modify with face data
- **negative**: Negative conditioning to modify with face data
- **ref_image**: Reference image containing the target face
- **latent_image**: Latent representation for generation
- **adapter_file**: InfiniteYou adapter model file
- **weight**: Strength of the face application effect (0.0-5.0)
- **start_at**: Starting point in the diffusion process (0.0-1.0)
- **end_at**: Ending point in the diffusion process (0.0-1.0)
- **vae**: VAE model for encoding/decoding images
- **fixed_face_pose**: Whether to maintain the exact pose from the reference image

## Outputs
- **MODEL**: The model with face settings applied
- **positive**: Positive conditioning with face embeddings
- **negative**: Negative conditioning with face embeddings
- **latent**: Latent representation for generation
- **patch_data**: Face patch data that can be saved with the Patch Saver node

## Usage
1. Connect a ControlNet model designed for face or pose control
2. Provide a reference image containing a clear face
3. Adjust the weight, timing, and other parameters to control the face application effect
4. Optionally enable fixed_face_pose to maintain the exact pose from the reference
5. Connect the outputs to a sampler node to generate images with the target face
6. Connect the patch_data output to a Star InfiniteYou Patch Saver to save the face for future use

## Features

### Direct Face Extraction
- Automatically detects and processes faces in reference images
- Extracts facial landmarks for precise feature positioning
- Creates face embeddings that capture the identity and characteristics

### Flexible Application Control
- Adjustable weight for controlling how strongly the face is applied
- Start/end parameters to control when in the diffusion process the face is applied
- Fixed pose option to maintain the exact facial orientation from the reference

### Patch Data Generation
- Creates reusable face patch data that can be saved for future use
- Includes all necessary information for consistent face application
- Compatible with the Star InfiniteYou Patch Saver node

### Advanced Face Processing
- Applies face information through ControlNet conditioning
- Processes both positive and negative conditioning for consistent results
- Handles noise addition for more natural integration
- Automatically resizes face data to match target dimensions

## Technical Details
- Uses the InfiniteYou model to extract facial features and embeddings
- Applies face information through ControlNet conditioning
- Processes both positive and negative conditioning for consistent results
- Automatically handles device and dtype management for compatibility
- Supports both fixed and adaptive face pose application
- Built-in noise parameter (0.35) for natural variation

## Notes
- For best results, use reference images with clear, front-facing faces
- Higher weight values produce stronger resemblance to the reference face
- The start_at and end_at parameters can be adjusted to balance identity preservation and image quality
- The fixed_face_pose option is useful when you want to maintain the exact head position and angle
- This node works best with models that have good face generation capabilities
- Save the patch_data output with the Star InfiniteYou Patch Saver for reuse in future workflows
