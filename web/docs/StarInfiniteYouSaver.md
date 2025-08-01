# Star InfiniteYou Patch Saver

## Description
The Star InfiniteYou Patch Saver is a utility node that saves face data (patches) generated by other InfiniteYou nodes for future use. This node allows you to preserve facial embeddings, landmarks, and control parameters, making it possible to consistently apply the same face across multiple workflows without needing the original reference images.

## Inputs

### Required
- **patch_data**: Face patch data generated by InfiniteYou nodes (typically from the Star InfiniteYou Face Swap Mod or Star InfiniteYou Patch Combine nodes)
- **save_name**: Name to use when saving the patch file (alphanumeric characters, underscores, and hyphens only)

## Outputs
This node has no outputs as it is designed solely for saving data.

## Usage
1. Connect the "patch_data" output from a Star InfiniteYou Face Swap Mod or Star InfiniteYou Patch Combine node
2. Enter a descriptive name for the face patch in the "save_name" field
3. Run the workflow to save the patch
4. The saved patch will be available for selection in other InfiniteYou nodes

## Features

### Persistent Storage
- Saves face patches to the "infiniteyoupatch" directory in the ComfyUI output folder
- Creates standardized .iyou files for compatibility across nodes
- Automatically sanitizes filenames to ensure validity

### Data Processing
- Performs deep copying to preserve original data
- Makes conditioning data serializable by removing non-serializable objects
- Moves tensors to CPU for efficient storage
- Includes fallback mechanisms for reliable saving

### Comprehensive Data Preservation
- Saves facial landmarks for precise feature positioning
- Preserves face embeddings that capture identity characteristics
- Stores control parameters (strength, timing) for consistent application
- Optionally saves processed conditioning when possible

## Technical Details
- Files are saved with the .iyou extension in the "infiniteyoupatch" directory
- The node handles tensor detachment and CPU conversion automatically
- Implements a two-tier saving system with fallback to essential data if full saving fails
- Sanitizes filenames to remove invalid characters
- Provides detailed console feedback during the saving process

## Notes
- Saved patches can be loaded using the Star InfiniteYou Patch Loader node
- Multiple patches can be combined using the Star InfiniteYou Patch Combine node
- For organization, use descriptive names that identify the face or its characteristics
- Patches are saved in the ComfyUI output directory under the "infiniteyoupatch" folder
- The node will automatically create the required directories if they don't exist
