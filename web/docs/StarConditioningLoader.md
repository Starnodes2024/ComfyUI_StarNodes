# Star Conditioning Loader

## Description
The Star Conditioning Loader node allows you to load previously saved conditioning data from files. This node complements the Star Conditioning Saver node, enabling you to reuse conditioning setups across different workflows, share them with others, or maintain a library of conditioning presets. It automatically detects and lists available conditioning files, making it easy to select and load the desired conditioning data.

## Inputs

### Required
- **conditioning_file**: Dropdown selection of available conditioning files (sorted by newest first)

## Outputs
- **conditioning**: The loaded conditioning data, ready to be used in a workflow

## Usage
1. Add the Star Conditioning Loader node to your workflow
2. Select a previously saved conditioning file from the dropdown menu
3. Run the node to load the conditioning data
4. Connect the conditioning output to nodes that require conditioning input, such as samplers

## Features
- **Automatic File Detection**: Automatically scans the conditionings directory and populates the dropdown with available files
- **Chronological Sorting**: Files are sorted by modification time with newest files first for easy access
- **Graceful Error Handling**: Provides meaningful feedback if files can't be loaded or if no files are available
- **Full Conditioning Restoration**: Loads the complete conditioning data structure exactly as it was saved

## Technical Details
- Loads conditioning files from the `[ComfyUI output directory]/conditionings/` folder
- Compatible with files saved by the Star Conditioning Saver node
- Files are expected to be in PyTorch's .pt format
- If no conditioning files are found, the dropdown will show "No conditioning files found" and the node will output an empty conditioning list

## Notes
- This node works in tandem with the Star Conditioning Saver node
- For best results, use descriptive filenames when saving conditioning data
- The loaded conditioning can be used directly with any node that accepts conditioning inputs
- This node is particularly useful for:
  - Maintaining a library of frequently used prompt conditioning
  - Sharing specific conditioning setups with others
  - Creating consistent results across different workflow sessions
  - Quickly switching between different conditioning presets without having to recreate complex prompt structures
