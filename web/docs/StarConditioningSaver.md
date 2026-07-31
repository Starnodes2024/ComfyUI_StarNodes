# Star Conditioning Saver

## Description
The Star Conditioning Saver node allows you to save conditioning data to a file for later use. This is particularly useful for preserving complex conditioning setups, sharing them between workflows, or creating libraries of conditioning presets. The node saves the conditioning in a portable format that can be loaded back into ComfyUI using the Star Conditioning Loader node.

## Inputs

### Required
- **conditioning**: The conditioning data to save (typically from a CLIP Text Encode node or other conditioning source)
- **filename**: Base name for the saved file (default: "conditioning")

## Outputs
- **conditioning**: The same conditioning data that was input (passed through unchanged)

## Usage
1. Connect the conditioning output from a CLIP Text Encode node or other conditioning source to the conditioning input
2. Enter a descriptive filename for your conditioning data
3. Run the node to save the conditioning data
4. The conditioning will be saved to the ComfyUI output directory under a "conditionings" subfolder

## Features
- **Automatic Timestamping**: Adds a timestamp to filenames to prevent overwriting previous saves
- **Filename Sanitization**: Ensures filenames are valid by removing problematic characters
- **Tensor Serialization**: Properly handles the serialization of tensor data for reliable storage
- **Metadata Storage**: Saves additional metadata such as timestamp and original filename
- **Pass-through Design**: Returns the original conditioning, allowing the node to be inserted anywhere in a workflow without disrupting the flow

## Technical Details
- Files are saved in PyTorch's .pt format in the `[ComfyUI output directory]/conditionings/` folder
- Each file contains both the conditioning tensors and metadata
- Filenames follow the pattern `[your_filename]_[YYYYMMDD_HHMMSS].pt`
- The node handles the conversion of tensor data to a format suitable for saving

## Notes
- Saved conditioning files can be loaded using the Star Conditioning Loader node
- The conditioning data structure is preserved exactly as it was when saved
- This node is useful for creating libraries of prompt conditioning that can be reused across different workflows
- For best organization, use descriptive filenames that indicate the content or purpose of the conditioning
